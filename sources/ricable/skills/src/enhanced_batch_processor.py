#!/usr/bin/env python3
"""
Enhanced Scalable Batch Processing System for Ericsson Documentation
Implements memory-efficient processing for large datasets (2000+ files)
With resume capability, error recovery, and configurable batching
"""

import os
import sys
import json
import re
import hashlib
import time
import gc
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Iterator
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BatchState:
    """State management for batch processing"""
    batch_num: int
    total_batches: int
    files_in_batch: int
    processed_files: List[str]
    failed_files: List[Tuple[str, str]]
    start_time: float
    end_time: Optional[float] = None
    memory_usage: Optional[float] = None
    status: str = "pending"  # pending, processing, completed, failed


@dataclass
class ProcessingStats:
    """Comprehensive processing statistics"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    cached_files: int = 0
    features_extracted: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    batches_completed: int = 0
    average_batch_time: float = 0.0
    memory_peak: float = 0.0
    errors: List[Tuple[str, str]] = field(default_factory=list)


class BatchProcessor:
    """
    Enhanced batch processing system with memory management and resume capability
    """

    def __init__(self,
                 source_dir: str,
                 output_dir: str = "output",
                 batch_size: int = 50,
                 max_memory_mb: float = 1024.0,
                 auto_gc: bool = True,
                 resume: bool = True):
        """
        Initialize batch processor

        Args:
            source_dir: Directory containing markdown files
            output_dir: Output directory for processed data
            batch_size: Number of files per batch (default: 50)
            max_memory_mb: Maximum memory usage in MB before forced cleanup
            auto_gc: Enable automatic garbage collection
            resume: Enable resume capability from last successful batch
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.auto_gc = auto_gc
        self.resume_enabled = resume

        # Processing state
        self.batch_states: List[BatchState] = []
        self.current_batch = 0
        self.stats = ProcessingStats()

        # Data storage (will be cleared between batches to manage memory)
        self.features_batch: Dict[str, any] = {}
        self.processed_files_batch: Set[str] = set()

        # Setup directories
        self.setup_directories()

        # Load previous state if resuming
        if self.resume_enabled:
            self.load_progress()

    def setup_directories(self):
        """Create necessary output directories"""
        directories = [
            self.output_dir / "ericsson_data",
            self.output_dir / "ericsson_data" / "features",
            self.output_dir / "ericsson_data" / "cache",
            self.output_dir / "ericsson_data" / "indices",
            self.output_dir / "ericsson_data" / "logs"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def discover_files(self, pattern: str = "*.md") -> List[Path]:
        """
        Discover all files matching the pattern using glob patterns

        Args:
            pattern: Glob pattern for file discovery

        Returns:
            List of file paths to process
        """
        logger.info(f"🔍 Discovering files with pattern '{pattern}' in {self.source_dir}")

        # Use rglob for recursive search
        files = list(self.source_dir.rglob(pattern))

        # Filter out already processed files if resuming
        if self.resume_enabled and self.batch_states:
            processed_files = set()
            for batch_state in self.batch_states:
                if batch_state.status == "completed":
                    processed_files.update(batch_state.processed_files)

            if processed_files:
                files = [f for f in files if str(f) not in processed_files]
                logger.info(f"📋 Resuming: {len(files)} files remaining after skipping {len(processed_files)} already processed")

        self.stats.total_files = len(files)
        logger.info(f"📊 Found {len(files)} files to process")

        return files

    def create_file_batches(self, files: List[Path]) -> Iterator[List[Path]]:
        """
        Create file batches with memory-efficient iteration

        Args:
            files: List of files to batch

        Yields:
            Batches of file paths
        """
        total_batches = (len(files) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.info(f"📦 Creating batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            yield batch_files, batch_num, total_batches

    def check_memory_usage(self) -> float:
        """
        Check current memory usage in MB

        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            # If psutil not available, estimate based on Python's memory
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            current, peak = tracemalloc.get_traced_memory()
            return peak / 1024 / 1024

    def force_cleanup(self):
        """Force garbage collection and cleanup to free memory"""
        logger.info("🧹 Forcing memory cleanup...")

        # Clear batch-specific data
        self.features_batch.clear()
        self.processed_files_batch.clear()

        # Force garbage collection
        gc.collect()

        # Check memory after cleanup
        memory_after = self.check_memory_usage()
        logger.info(f"✅ Memory after cleanup: {memory_after:.1f} MB")

    def process_batch(self, files: List[Path], batch_num: int, total_batches: int) -> BatchState:
        """
        Process a single batch of files with error handling and memory management

        Args:
            files: List of files in this batch
            batch_num: Current batch number
            total_batches: Total number of batches

        Returns:
            BatchState with processing results
        """
        batch_state = BatchState(
            batch_num=batch_num,
            total_batches=total_batches,
            files_in_batch=len(files),
            processed_files=[],
            failed_files=[],
            start_time=time.time(),
            status="processing"
        )

        logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(files)} files)")

        # Check memory before batch
        memory_before = self.check_memory_usage()
        logger.debug(f"Memory before batch: {memory_before:.1f} MB")

        for i, file_path in enumerate(files):
            try:
                # Process individual file
                result = self.process_file(file_path)

                if result:
                    # Store in batch storage
                    self.features_batch[result.get('id')] = result
                    self.processed_files_batch.add(str(file_path))
                    batch_state.processed_files.append(str(file_path))

                    logger.debug(f"✅ Processed {file_path.name} ({i+1}/{len(files)})")
                else:
                    batch_state.failed_files.append((str(file_path), "No valid feature extracted"))
                    logger.warning(f"⚠️ No feature extracted from {file_path.name}")

                # Progress indicator every 10 files in batch
                if (i + 1) % 10 == 0:
                    logger.info(f"  Batch progress: {i+1}/{len(files)} files processed")

                # Check memory usage and cleanup if needed
                if self.auto_gc and (i + 1) % 25 == 0:
                    current_memory = self.check_memory_usage()
                    if current_memory > self.max_memory_mb:
                        logger.warning(f"Memory usage ({current_memory:.1f} MB) exceeds limit ({self.max_memory_mb} MB)")
                        self.force_cleanup()

            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                batch_state.failed_files.append((str(file_path), str(e)))
                self.stats.errors.append((str(file_path), str(e)))
                logger.error(f"❌ {error_msg}")

        # Complete batch processing
        batch_state.end_time = time.time()
        batch_state.memory_usage = self.check_memory_usage()
        batch_state.status = "completed"

        # Save batch results immediately
        self.save_batch_results(batch_state)

        # Update statistics
        self.update_batch_statistics(batch_state)

        # Cleanup batch data to free memory
        self.force_cleanup()

        logger.info(f"✅ Batch {batch_num} completed in {batch_state.end_time - batch_state.start_time:.1f}s")
        logger.info(f"   Processed: {len(batch_state.processed_files)}, Failed: {len(batch_state.failed_files)}")

        return batch_state

    def process_file(self, file_path: Path) -> Optional[Dict]:
        """
        Process a single file - to be implemented by specific processor

        Args:
            file_path: Path to file to process

        Returns:
            Processed data or None if processing failed
        """
        # This is a placeholder - actual implementation would be in the specific processor
        # For now, just simulate processing
        time.sleep(0.01)  # Simulate processing time

        # Return mock data
        return {
            'id': f"feature_{file_path.stem}",
            'source_file': str(file_path),
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def save_batch_results(self, batch_state: BatchState):
        """
        Save results from completed batch

        Args:
            batch_state: State of completed batch
        """
        # Save individual feature files
        features_dir = self.output_dir / "ericsson_data" / "features"
        for feature_id, feature_data in self.features_batch.items():
            filename = f"feature_{feature_id.replace(' ', '_')}.json"
            filepath = features_dir / filename
            filepath.write_text(json.dumps(feature_data, indent=2))

        # Save batch state
        batch_file = self.output_dir / "ericsson_data" / "logs" / f"batch_{batch_state.batch_num:04d}.json"
        batch_file.write_text(json.dumps(asdict(batch_state), indent=2))

    def update_batch_statistics(self, batch_state: BatchState):
        """
        Update processing statistics with batch results

        Args:
            batch_state: Completed batch state
        """
        self.stats.processed_files += len(batch_state.processed_files)
        self.stats.failed_files += len(batch_state.failed_files)
        self.stats.features_extracted += len(self.features_batch)
        self.stats.batches_completed += 1

        # Update average batch time
        batch_duration = batch_state.end_time - batch_state.start_time
        if self.stats.batches_completed == 1:
            self.stats.average_batch_time = batch_duration
        else:
            self.stats.average_batch_time = (
                (self.stats.average_batch_time * (self.stats.batches_completed - 1) + batch_duration) /
                self.stats.batches_completed
            )

        # Update peak memory
        if batch_state.memory_usage and batch_state.memory_usage > self.stats.memory_peak:
            self.stats.memory_peak = batch_state.memory_usage

    def save_progress(self):
        """Save current processing progress for resume capability"""
        progress_data = {
            'stats': asdict(self.stats),
            'batch_states': [asdict(state) for state in self.batch_states],
            'current_batch': self.current_batch,
            'resume_enabled': self.resume_enabled,
            'config': {
                'batch_size': self.batch_size,
                'max_memory_mb': self.max_memory_mb,
                'auto_gc': self.auto_gc
            }
        }

        progress_file = self.output_dir / "ericsson_data" / "progress.json"
        progress_file.write_text(json.dumps(progress_data, indent=2, default=str))

        logger.info(f"💾 Progress saved after batch {self.current_batch}")

    def load_progress(self):
        """Load previous processing progress for resuming"""
        progress_file = self.output_dir / "ericsson_data" / "progress.json"

        if not progress_file.exists():
            logger.info("🆕 No previous progress found - starting fresh")
            return

        try:
            progress_data = json.loads(progress_file.read_text())

            # Restore batch states
            self.batch_states = [
                BatchState(**state_data)
                for state_data in progress_data.get('batch_states', [])
            ]

            # Restore statistics
            stats_data = progress_data.get('stats', {})
            self.stats = ProcessingStats(**stats_data)

            # Set current batch
            self.current_batch = progress_data.get('current_batch', 0)

            # Validate config consistency
            config = progress_data.get('config', {})
            if config.get('batch_size') != self.batch_size:
                logger.warning(f"Batch size changed from {config.get('batch_size')} to {self.batch_size}")

            logger.info(f"📋 Loaded progress: {self.stats.batches_completed} batches completed")
            logger.info(f"   Previously processed: {self.stats.processed_files} files")
            logger.info(f"   Failed: {self.stats.failed_files} files")

        except Exception as e:
            logger.error(f"❌ Error loading progress: {e}")
            logger.info("🆕 Starting fresh due to progress load error")

    def process_all(self, limit: Optional[int] = None, pattern: str = "*.md"):
        """
        Main processing method with enhanced batching and error handling

        Args:
            limit: Optional limit on number of files to process
            pattern: Glob pattern for file discovery
        """
        logger.info("🚀 Starting enhanced batch processing")
        logger.info(f"   Source: {self.source_dir}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Memory limit: {self.max_memory_mb} MB")
        logger.info(f"   Resume enabled: {self.resume_enabled}")

        start_time = time.time()

        try:
            # Discover files
            files = self.discover_files(pattern)

            if not files:
                logger.info("✅ No files to process")
                return

            # Apply limit if specified
            if limit:
                files = files[:limit]
                logger.info(f"🎯 Processing limited to {limit} files")

            # Process in batches
            for batch_files, batch_num, total_batches in self.create_file_batches(files):
                self.current_batch = batch_num

                # Process batch
                batch_state = self.process_batch(batch_files, batch_num, total_batches)
                self.batch_states.append(batch_state)

                # Save progress every 5 batches
                if batch_num % 5 == 0:
                    self.save_progress()

                # Check for memory cleanup after each batch
                if self.auto_gc:
                    current_memory = self.check_memory_usage()
                    if current_memory > self.max_memory_mb * 0.8:  # 80% threshold
                        logger.info(f"Memory usage ({current_memory:.1f} MB) approaching limit")
                        self.force_cleanup()

            # Final processing
            self.finalize_processing()

        except KeyboardInterrupt:
            logger.info("\n⚠️ Processing interrupted by user")
            self.save_progress()
            logger.info("💾 Progress saved - you can resume by running with --resume")

        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            self.save_progress()
            raise

        finally:
            # Final statistics
            end_time = time.time()
            total_time = end_time - start_time
            self.stats.end_time = end_time

            self.print_final_summary(total_time)

    def finalize_processing(self):
        """Finalize processing after all batches complete"""
        logger.info("🏁 Finalizing processing...")

        # Save final progress
        self.save_progress()

        # Build final indices if needed (placeholder)
        logger.info("📊 Building search indices...")

        # Create final summary
        self.create_processing_summary()

        logger.info("✅ Processing finalized")

    def create_processing_summary(self):
        """Create comprehensive processing summary"""
        summary = {
            'processing_stats': asdict(self.stats),
            'batch_summary': [
                {
                    'batch_num': state.batch_num,
                    'files_processed': len(state.processed_files),
                    'files_failed': len(state.failed_files),
                    'duration': state.end_time - state.start_time if state.end_time else 0,
                    'memory_usage': state.memory_usage,
                    'status': state.status
                }
                for state in self.batch_states
            ],
            'performance_metrics': {
                'files_per_second': self.stats.processed_files / (self.stats.end_time - self.stats.start_time) if self.stats.end_time else 0,
                'average_batch_time': self.stats.average_batch_time,
                'peak_memory_mb': self.stats.memory_peak,
                'success_rate': (self.stats.processed_files / max(self.stats.total_files, 1)) * 100
            },
            'configuration': {
                'batch_size': self.batch_size,
                'max_memory_mb': self.max_memory_mb,
                'auto_gc': self.auto_gc,
                'resume_enabled': self.resume_enabled
            }
        }

        summary_file = self.output_dir / "ericsson_data" / "processing_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2, default=str))

        logger.info(f"📄 Processing summary saved to {summary_file}")

    def print_final_summary(self, total_time: float):
        """Print comprehensive final processing summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files: {self.stats.total_files}")
        logger.info(f"Processed: {self.stats.processed_files}")
        logger.info(f"Failed: {self.stats.failed_files}")
        logger.info(f"Features extracted: {self.stats.features_extracted}")
        logger.info(f"Batches completed: {self.stats.batches_completed}")
        logger.info(f"Total time: {total_time:.1f} seconds")

        if self.stats.processed_files > 0:
            logger.info(f"Average speed: {self.stats.processed_files/total_time:.1f} files/second")

        logger.info(f"Peak memory usage: {self.stats.memory_peak:.1f} MB")
        logger.info(f"Success rate: {(self.stats.processed_files/max(self.stats.total_files,1))*100:.1f}%")

        if self.stats.errors:
            logger.info(f"Errors encountered: {len(self.stats.errors)}")
            # Show first few errors
            for i, (file, error) in enumerate(self.stats.errors[:3]):
                logger.info(f"  - {Path(file).name}: {error}")
            if len(self.stats.errors) > 3:
                logger.info(f"  ... and {len(self.stats.errors) - 3} more errors")

        logger.info("="*60)


def main():
    """Example usage of the batch processing system"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Batch Processing System")
    parser.add_argument("--source", required=True, help="Source directory with files")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--max-memory", type=float, default=1024, help="Max memory in MB")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--pattern", default="*.md", help="File pattern to match")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume capability")
    parser.add_argument("--no-gc", action="store_true", help="Disable automatic garbage collection")

    args = parser.parse_args()

    # Create processor
    processor = BatchProcessor(
        source_dir=args.source,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_memory_mb=args.max_memory,
        auto_gc=not args.no_gc,
        resume=not args.no_resume
    )

    # Start processing
    processor.process_all(limit=args.limit, pattern=args.pattern)


if __name__ == "__main__":
    main()