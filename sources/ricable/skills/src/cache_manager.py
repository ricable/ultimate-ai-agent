#!/usr/bin/env python3
"""
File-based caching system with MD5 validation for Ericsson feature processing.
Provides robust cache management with incremental updates and hash validation.
"""

import os
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached feature with metadata"""
    file_path: str
    md5_hash: str
    processed_time: float
    feature_data: Dict[str, Any]
    file_size: int
    processing_time: float  # Time taken to process the file

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary for JSON deserialization"""
        return cls(**data)


class CacheManager:
    """
    Manages file-based caching with MD5 validation for Ericsson feature processing.

    Features:
    - MD5 hash validation to detect file changes
    - Incremental processing by skipping unchanged files
    - Batch-based cache saving for performance
    - Cache cleanup and invalidation
    - Statistics tracking
    - Error handling and recovery
    """

    def __init__(self, cache_dir: str = "output/ericsson_data/cache"):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_stats_file = self.cache_dir / "cache_stats.json"

        # In-memory cache index
        self.cache_index: Dict[str, CacheEntry] = {}
        self.dirty_entries: Set[str] = set()  # Modified entries to save

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'files_processed': 0,
            'files_skipped': 0,
            'total_processing_time': 0.0,
            'cache_size_mb': 0.0,
            'last_cleanup': time.time()
        }

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache
        self._load_cache_index()
        self._load_stats()

        logger.info(f"Cache manager initialized with {len(self.cache_index)} cached entries")

    def _generate_md5(self, file_path: str) -> str:
        """
        Generate MD5 hash for a file

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash as hexadecimal string
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate MD5 for {file_path}: {e}")
            raise

    def _get_cache_key(self, file_path: str) -> str:
        """
        Generate a cache key for a file path

        Args:
            file_path: Original file path

        Returns:
            Normalized cache key
        """
        # Use relative path from cache directory or absolute path
        return str(Path(file_path).as_posix())

    def _load_cache_index(self):
        """Load cache index from disk"""
        if not self.cache_index_file.exists():
            logger.info("No existing cache index found")
            return

        try:
            with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct cache entries
            for cache_key, entry_data in data.items():
                try:
                    self.cache_index[cache_key] = CacheEntry.from_dict(entry_data)
                except Exception as e:
                    logger.warning(f"Failed to load cache entry for {cache_key}: {e}")
                    continue

            logger.info(f"Loaded {len(self.cache_index)} cache entries from disk")

        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")
            # Start with empty cache if loading fails
            self.cache_index = {}

    def _save_cache_index(self):
        """Save cache index to disk"""
        if not self.dirty_entries:
            return  # Nothing to save

        try:
            # Convert cache entries to dictionaries
            data = {}
            for cache_key, entry in self.cache_index.items():
                if cache_key in self.dirty_entries:
                    data[cache_key] = entry.to_dict()
                else:
                    # Load existing data for unchanged entries
                    if self.cache_index_file.exists():
                        try:
                            with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                                existing_data = json.load(f)
                                if cache_key in existing_data:
                                    data[cache_key] = existing_data[cache_key]
                        except:
                            pass

                    # Add current entry if not in existing data
                    if cache_key not in data:
                        data[cache_key] = entry.to_dict()

            # Write to temporary file first, then move to avoid corruption
            temp_file = self.cache_index_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic move
            temp_file.replace(self.cache_index_file)

            self.dirty_entries.clear()
            logger.debug(f"Saved cache index with {len(data)} entries")

        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
            raise

    def _load_stats(self):
        """Load cache statistics from disk"""
        if not self.cache_stats_file.exists():
            return

        try:
            with open(self.cache_stats_file, 'r', encoding='utf-8') as f:
                loaded_stats = json.load(f)
                self.stats.update(loaded_stats)
        except Exception as e:
            logger.warning(f"Failed to load cache stats: {e}")

    def _save_stats(self):
        """Save cache statistics to disk"""
        try:
            # Calculate current cache size
            self.stats['cache_size_mb'] = self._get_cache_size_mb()
            self.stats['cached_files'] = len(self.cache_index)

            temp_file = self.cache_stats_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)

            temp_file.replace(self.cache_stats_file)

        except Exception as e:
            logger.warning(f"Failed to save cache stats: {e}")

    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB"""
        try:
            total_size = 0
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    def is_cached(self, file_path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a file is cached and return cached feature data

        Args:
            file_path: Path to the source file

        Returns:
            Tuple of (is_cached, feature_data or None)
        """
        cache_key = self._get_cache_key(file_path)

        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Source file not found: {file_path}")
            return False, None

        try:
            # Generate current MD5 hash
            current_hash = self._generate_md5(file_path)
            current_size = os.path.getsize(file_path)

            # Check if entry exists in cache
            if cache_key in self.cache_index:
                cached_entry = self.cache_index[cache_key]

                # Validate hash and file size
                if (cached_entry.md5_hash == current_hash and
                    cached_entry.file_size == current_size):
                    self.stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for {file_path}")
                    return True, cached_entry.feature_data
                else:
                    # File has changed, remove old entry
                    logger.debug(f"File changed, removing cache entry for {file_path}")
                    del self.cache_index[cache_key]
                    self.dirty_entries.discard(cache_key)

            self.stats['cache_misses'] += 1
            return False, None

        except Exception as e:
            logger.error(f"Error checking cache for {file_path}: {e}")
            return False, None

    def cache_feature(self, file_path: str, feature_data: Dict[str, Any], processing_time: float):
        """
        Cache a processed feature

        Args:
            file_path: Path to the source file
            feature_data: Processed feature data
            processing_time: Time taken to process the file
        """
        try:
            cache_key = self._get_cache_key(file_path)
            current_hash = self._generate_md5(file_path)
            file_size = os.path.getsize(file_path)

            # Create cache entry
            entry = CacheEntry(
                file_path=file_path,
                md5_hash=current_hash,
                processed_time=time.time(),
                feature_data=feature_data,
                file_size=file_size,
                processing_time=processing_time
            )

            # Add to cache
            self.cache_index[cache_key] = entry
            self.dirty_entries.add(cache_key)

            self.stats['files_processed'] += 1
            self.stats['total_processing_time'] += processing_time

            logger.debug(f"Cached feature for {file_path}")

        except Exception as e:
            logger.error(f"Failed to cache feature for {file_path}: {e}")

    def save_batch(self):
        """Save cache index for batch processing"""
        try:
            self._save_cache_index()
            self._save_stats()
            logger.debug("Cache batch saved successfully")
        except Exception as e:
            logger.error(f"Failed to save cache batch: {e}")

    def clear_cache(self):
        """Clear all cached data"""
        try:
            # Remove cache directory contents
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Reset in-memory data
            self.cache_index.clear()
            self.dirty_entries.clear()
            self.stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'files_processed': 0,
                'files_skipped': 0,
                'total_processing_time': 0.0,
                'cache_size_mb': 0.0,
                'last_cleanup': time.time()
            }

            logger.info("Cache cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise

    def cleanup_invalid_entries(self):
        """Remove cache entries for files that no longer exist"""
        try:
            invalid_keys = []

            for cache_key, entry in self.cache_index.items():
                if not os.path.exists(entry.file_path):
                    invalid_keys.append(cache_key)

            # Remove invalid entries
            for key in invalid_keys:
                del self.cache_index[key]
                self.dirty_entries.discard(key)

            if invalid_keys:
                self._save_cache_index()
                logger.info(f"Removed {len(invalid_keys)} invalid cache entries")

        except Exception as e:
            logger.error(f"Failed to cleanup invalid entries: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        return {
            'cache_directory': str(self.cache_dir),
            'cached_files': len(self.cache_index),
            'cache_hits': self.stats.get('cache_hits', 0),
            'cache_misses': self.stats.get('cache_misses', 0),
            'files_processed': self.stats.get('files_processed', 0),
            'files_skipped': self.stats.get('files_skipped', 0),
            'cache_size_mb': round(self._get_cache_size_mb(), 2),
            'hit_rate': self._calculate_hit_rate(),
            'last_cleanup': self.stats.get('last_cleanup', 0)
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage"""
        hits = self.stats.get('cache_hits', 0)
        misses = self.stats.get('cache_misses', 0)
        total = hits + misses

        if total == 0:
            return 0.0

        return round((hits / total) * 100, 2)

    def list_cached_files(self) -> List[str]:
        """List all cached file paths"""
        return [entry.file_path for entry in self.cache_index.values()]

    def get_cache_entry(self, file_path: str) -> Optional[CacheEntry]:
        """Get cache entry for a specific file"""
        cache_key = self._get_cache_key(file_path)
        return self.cache_index.get(cache_key)

    def remove_cache_entry(self, file_path: str):
        """Remove cache entry for a specific file"""
        cache_key = self._get_cache_key(file_path)
        if cache_key in self.cache_index:
            del self.cache_index[cache_key]
            self.dirty_entries.discard(cache_key)
            logger.debug(f"Removed cache entry for {file_path}")

    def force_reprocess(self, file_path: str):
        """Force reprocessing of a file by removing its cache entry"""
        self.remove_cache_entry(file_path)
        logger.info(f"Forced reprocessing for {file_path}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save any pending changes"""
        try:
            self.save_batch()
            logger.info("Cache manager shutdown completed")
        except Exception as e:
            logger.error(f"Error during cache manager shutdown: {e}")


def main():
    """Test the cache manager functionality"""
    import tempfile

    # Create a temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "test_cache")

        # Test cache manager
        with CacheManager(cache_dir) as cache:
            print("Cache Manager Test")
            print("=" * 50)

            # Create test file
            test_file = os.path.join(temp_dir, "test_feature.md")
            with open(test_file, 'w') as f:
                f.write("# Test Feature\nFAJ 123 4567\nTest content")

            # Test cache miss
            is_cached, data = cache.is_cached(test_file)
            print(f"First check - Cached: {is_cached}, Data: {data}")

            # Cache test data
            test_feature = {
                'id': 'FAJ 123 4567',
                'name': 'Test Feature',
                'description': 'Test description'
            }
            cache.cache_feature(test_file, test_feature, processing_time=0.5)

            # Test cache hit
            is_cached, data = cache.is_cached(test_file)
            print(f"Second check - Cached: {is_cached}, Data: {data}")

            # Get cache info
            info = cache.get_cache_info()
            print(f"Cache info: {info}")

            # Test file change detection
            with open(test_file, 'w') as f:
                f.write("# Modified Test Feature\nFAJ 123 4567\nModified content")

            is_cached, data = cache.is_cached(test_file)
            print(f"After modification - Cached: {is_cached}, Data: {data}")

            print("\nTest completed successfully!")


if __name__ == "__main__":
    main()