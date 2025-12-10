#!/usr/bin/env python3
"""
Ericsson Feature Processor with integrated caching system.
Combines the feature processing logic with robust file-based caching.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Import our modules
from cache_manager import CacheManager
from ericsson_feature_processor import EricssonFeatureProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CachedFeatureProcessor:
    """
    Ericsson Feature Processor with integrated caching system.
    Provides incremental processing with MD5 validation.
    """

    def __init__(self, cache_dir: str = "output/ericsson_data/cache"):
        """
        Initialize the cached feature processor

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_manager = CacheManager(cache_dir)
        # Initialize with dummy values since we'll use individual file processing
        self.processor = EricssonFeatureProcessor(".", ".")
        self.processing_stats = {
            'files_scanned': 0,
            'files_from_cache': 0,
            'files_processed': 0,
            'files_failed': 0,
            'start_time': time.time(),
            'total_processing_time': 0.0,
            'cache_time_saved': 0.0
        }

        logger.info("Cached Feature Processor initialized")

    def process_file_with_cache(self, file_path: str) -> Optional[Dict]:
        """
        Process a single file with caching support

        Args:
            file_path: Path to the markdown file

        Returns:
            Processed feature data or None if processing failed
        """
        self.processing_stats['files_scanned'] += 1

        # Check cache first
        is_cached, cached_data = self.cache_manager.is_cached(file_path)

        if is_cached and cached_data:
            self.processing_stats['files_from_cache'] += 1
            logger.info(f"Using cached data for {os.path.basename(file_path)}")
            return cached_data

        # Process file if not cached
        logger.info(f"Processing {os.path.basename(file_path)}")
        start_time = time.time()

        try:
            # Use the existing processor to extract feature data
            feature_data = self._extract_feature_data(file_path)

            if feature_data:
                processing_time = time.time() - start_time
                self.processing_stats['total_processing_time'] += processing_time

                # Cache the processed data
                self.cache_manager.cache_feature(file_path, feature_data, processing_time)
                self.processing_stats['files_processed'] += 1

                logger.info(f"Processed and cached {os.path.basename(file_path)} in {processing_time:.2f}s")
                return feature_data
            else:
                logger.warning(f"No feature data extracted from {file_path}")
                self.processing_stats['files_failed'] += 1
                return None

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.processing_stats['files_failed'] += 1
            return None

    def _extract_feature_data(self, file_path: str) -> Optional[Dict]:
        """
        Extract feature data using the existing processor logic

        Args:
            file_path: Path to the markdown file

        Returns:
            Feature data dictionary or None
        """
        try:
            # Read and parse the markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Convert to BeautifulSoup for processing
            import markdown
            from bs4 import BeautifulSoup
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')

            # Extract feature identity
            feature = self.processor.extract_feature_identity(soup)
            if not feature.id:
                logger.debug(f"No FAJ ID found in {file_path}")
                return None

            # Extract all other components
            feature.parameters = self.processor.extract_parameters(soup)
            feature.counters = self.processor.extract_counters(soup)
            feature.events = self.processor.extract_events(soup)
            feature.dependencies = self.processor.extract_dependencies(soup)
            feature.activation_step = self.processor.extract_activation_step(soup)
            feature.deactivation_step = self.processor.extract_deactivation_step(soup)
            feature.engineering_guidelines = self.processor.extract_engineering_guidelines(soup)

            # Convert to dictionary for caching
            return {
                'id': feature.id,
                'name': feature.name,
                'cxc_code': feature.cxc_code,
                'value_package': feature.value_package,
                'value_package_id': feature.value_package_id,
                'access_type': feature.access_type,
                'node_type': feature.node_type,
                'description': feature.description,
                'summary': feature.summary,
                'parameters': feature.parameters,
                'counters': feature.counters,
                'events': feature.events,
                'dependencies': feature.dependencies,
                'activation_step': feature.activation_step,
                'deactivation_step': feature.deactivation_step,
                'engineering_guidelines': feature.engineering_guidelines
            }

        except Exception as e:
            logger.error(f"Error extracting feature data from {file_path}: {e}")
            return None

    def process_all_files(self, source_dir: str, batch_size: int = 50) -> List[Dict]:
        """
        Process all markdown files in source directory with caching

        Args:
            source_dir: Directory containing markdown files
            batch_size: Number of files to process before saving cache

        Returns:
            List of all processed features
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return []

        # Find all markdown files
        markdown_files = list(source_path.rglob("*.md"))
        logger.info(f"Found {len(markdown_files)} markdown files")

        if not markdown_files:
            logger.warning("No markdown files found")
            return []

        all_features = []
        processed_count = 0

        # Process files in batches
        for i in range(0, len(markdown_files), batch_size):
            batch_files = markdown_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_files)} files")

            batch_features = []
            for file_path in batch_files:
                feature_data = self.process_file_with_cache(str(file_path))
                if feature_data:
                    batch_features.append(feature_data)

            all_features.extend(batch_features)
            processed_count += len(batch_files)

            # Save cache after each batch
            self.cache_manager.save_batch()

            # Log progress
            cache_info = self.cache_manager.get_cache_info()
            logger.info(f"Batch completed: {len(batch_features)} features extracted")
            logger.info(f"Progress: {processed_count}/{len(markdown_files)} files")
            logger.info(f"Cache hit rate: {cache_info['hit_rate']:.1f}%")

        # Final cache save
        self.cache_manager.save_batch()

        # Calculate time saved by cache
        avg_processing_time = (self.processing_stats['total_processing_time'] /
                              max(1, self.processing_stats['files_processed']))
        self.processing_stats['cache_time_saved'] = (
            self.processing_stats['files_from_cache'] * avg_processing_time
        )

        return all_features

    def get_processing_report(self) -> Dict:
        """Generate comprehensive processing report"""
        total_time = time.time() - self.processing_stats['start_time']
        cache_info = self.cache_manager.get_cache_info()

        return {
            'processing_statistics': {
                'files_scanned': self.processing_stats['files_scanned'],
                'files_from_cache': self.processing_stats['files_from_cache'],
                'files_processed': self.processing_stats['files_processed'],
                'files_failed': self.processing_stats['files_failed'],
                'success_rate': round(
                    (self.processing_stats['files_processed'] /
                     max(1, self.processing_stats['files_scanned'])) * 100, 2
                ),
                'total_time_seconds': round(total_time, 2),
                'processing_time_seconds': round(self.processing_stats['total_processing_time'], 2),
                'cache_time_saved_seconds': round(self.processing_stats['cache_time_saved'], 2),
                'average_processing_time_seconds': round(
                    self.processing_stats['total_processing_time'] /
                    max(1, self.processing_stats['files_processed']), 2
                )
            },
            'cache_statistics': cache_info,
            'performance': {
                'files_per_second': round(
                    self.processing_stats['files_scanned'] / max(1, total_time), 2
                ),
                'cache_efficiency': round(
                    (self.processing_stats['cache_time_saved'] /
                     max(1, total_time)) * 100, 2
                )
            }
        }

    def force_reprocess_file(self, file_path: str):
        """Force reprocessing of a specific file"""
        self.cache_manager.force_reprocess(file_path)
        logger.info(f"Forced reprocessing for {file_path}")

    def force_reprocess_all(self, source_dir: str):
        """Force reprocessing of all files"""
        logger.info("Clearing cache and forcing reprocessing of all files")
        self.cache_manager.clear_cache()
        self.processing_stats = {
            'files_scanned': 0,
            'files_from_cache': 0,
            'files_processed': 0,
            'files_failed': 0,
            'start_time': time.time(),
            'total_processing_time': 0.0,
            'cache_time_saved': 0.0
        }

    def cleanup_cache(self):
        """Clean up invalid cache entries"""
        self.cache_manager.cleanup_invalid_entries()
        logger.info("Cache cleanup completed")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Ericsson Feature Processor with Caching")
    parser.add_argument("--source", default="elex_features_only",
                       help="Source directory containing markdown files")
    parser.add_argument("--output", default="output",
                       help="Output directory for processed data")
    parser.add_argument("--cache-dir", default="output/ericsson_data/cache",
                       help="Cache directory")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing")
    parser.add_argument("--limit", type=int,
                       help="Limit number of files to process")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing of all files")
    parser.add_argument("--cleanup-cache", action="store_true",
                       help="Clean up invalid cache entries")
    parser.add_argument("--cache-info", action="store_true",
                       help="Show cache information and exit")

    args = parser.parse_args()

    # Create cached processor
    processor = CachedFeatureProcessor(args.cache_dir)

    # Handle different operations
    if args.cache_info:
        info = processor.cache_manager.get_cache_info()
        print("Cache Information:")
        print(json.dumps(info, indent=2))
        return

    if args.cleanup_cache:
        processor.cleanup_cache()
        return

    if args.force_reprocess:
        processor.force_reprocess_all(args.source)

    # Process files
    try:
        with processor.cache_manager:
            features = processor.process_all_files(args.source, args.batch_size)

            # Save processed features
            output_dir = Path(args.output) / "ericsson_data"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save features
            features_file = output_dir / "features.json"
            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2, ensure_ascii=False)

            # Generate and save report
            report = processor.get_processing_report()
            report_file = output_dir / "processing_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Print summary
            print("\n" + "=" * 60)
            print("PROCESSING COMPLETED")
            print("=" * 60)
            print(f"Total files scanned: {report['processing_statistics']['files_scanned']}")
            print(f"Files from cache: {report['processing_statistics']['files_from_cache']}")
            print(f"Files processed: {report['processing_statistics']['files_processed']}")
            print(f"Files failed: {report['processing_statistics']['files_failed']}")
            print(f"Success rate: {report['processing_statistics']['success_rate']}%")
            print(f"Cache hit rate: {report['cache_statistics']['hit_rate']}%")
            print(f"Total time: {report['processing_statistics']['total_time_seconds']}s")
            print(f"Processing time: {report['processing_statistics']['processing_time_seconds']}s")
            print(f"Time saved by cache: {report['processing_statistics']['cache_time_saved_seconds']}s")
            print(f"Cache efficiency: {report['performance']['cache_efficiency']}%")
            print(f"\nFeatures saved to: {features_file}")
            print(f"Report saved to: {report_file}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        processor.cache_manager.save_batch()
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        processor.cache_manager.save_batch()
        sys.exit(1)


if __name__ == "__main__":
    main()