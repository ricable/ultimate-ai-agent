"""
Benchmark Database Module
=========================

This module provides SQLite database functionality for storing and retrieving
benchmark results, with support for historical tracking and performance analysis.
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from contextlib import contextmanager

from .core import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkDatabase:
    """SQLite database for storing benchmark results."""
    
    def __init__(self, db_path: str = "benchmarks/results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create benchmark_results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    cpu_usage REAL NOT NULL,
                    throughput REAL,
                    latency REAL,
                    error_rate REAL,
                    custom_metrics TEXT,  -- JSON string
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    metadata TEXT,  -- JSON string
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create benchmark_suites table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_suites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_name TEXT NOT NULL,
                    description TEXT,
                    total_benchmarks INTEGER NOT NULL,
                    successful_benchmarks INTEGER NOT NULL,
                    total_execution_time REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,  -- JSON string
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create performance_trends table for optimization tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    optimization_tag TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_benchmark_name_timestamp 
                ON benchmark_results(benchmark_name, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_suite_name_timestamp 
                ON benchmark_suites(suite_name, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trends_benchmark_metric 
                ON performance_trends(benchmark_name, metric_name, timestamp)
            """)
            
            conn.commit()
            logger.info(f"Database initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def store_result(self, result: BenchmarkResult) -> int:
        """Store a benchmark result in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO benchmark_results (
                    benchmark_name, timestamp, execution_time, memory_usage, cpu_usage,
                    throughput, latency, error_rate, custom_metrics, success, 
                    error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.benchmark_name,
                result.timestamp.isoformat(),
                result.metrics.execution_time,
                result.metrics.memory_usage,
                result.metrics.cpu_usage,
                result.metrics.throughput,
                result.metrics.latency,
                result.metrics.error_rate,
                json.dumps(result.metrics.custom_metrics),
                result.success,
                result.error_message,
                json.dumps(result.metadata)
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            
            # Store performance trends
            self._store_performance_trends(result)
            
            logger.debug(f"Stored benchmark result: {result.benchmark_name} (ID: {result_id})")
            return result_id
    
    def store_suite_results(self, suite_name: str, description: str, 
                          results: List[BenchmarkResult]) -> int:
        """Store benchmark suite results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Store individual results
            result_ids = []
            for result in results:
                result_id = self.store_result(result)
                result_ids.append(result_id)
            
            # Calculate suite metrics
            successful_benchmarks = sum(1 for r in results if r.success)
            total_execution_time = sum(r.metrics.execution_time for r in results)
            
            # Store suite summary
            cursor.execute("""
                INSERT INTO benchmark_suites (
                    suite_name, description, total_benchmarks, successful_benchmarks,
                    total_execution_time, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                suite_name,
                description,
                len(results),
                successful_benchmarks,
                total_execution_time,
                datetime.now().isoformat(),
                json.dumps({
                    'result_ids': result_ids,
                    'benchmark_names': [r.benchmark_name for r in results]
                })
            ))
            
            suite_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Stored suite results: {suite_name} ({len(results)} benchmarks, ID: {suite_id})")
            return suite_id
    
    def _store_performance_trends(self, result: BenchmarkResult) -> None:
        """Store performance trend data for optimization tracking."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Store key metrics as trends
            metrics_to_track = [
                ('execution_time', result.metrics.execution_time),
                ('memory_usage', result.metrics.memory_usage),
                ('cpu_usage', result.metrics.cpu_usage)
            ]
            
            if result.metrics.throughput is not None:
                metrics_to_track.append(('throughput', result.metrics.throughput))
            
            if result.metrics.latency is not None:
                metrics_to_track.append(('latency', result.metrics.latency))
            
            if result.metrics.error_rate is not None:
                metrics_to_track.append(('error_rate', result.metrics.error_rate))
            
            # Store custom metrics
            for metric_name, metric_value in result.metrics.custom_metrics.items():
                if isinstance(metric_value, (int, float)):
                    metrics_to_track.append((f'custom_{metric_name}', metric_value))
            
            # Insert trend data
            for metric_name, metric_value in metrics_to_track:
                cursor.execute("""
                    INSERT INTO performance_trends (
                        benchmark_name, metric_name, metric_value, timestamp
                    ) VALUES (?, ?, ?, ?)
                """, (
                    result.benchmark_name,
                    metric_name,
                    metric_value,
                    result.timestamp.isoformat()
                ))
            
            conn.commit()
    
    def get_latest_results(self, benchmark_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest results for a specific benchmark."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM benchmark_results 
                WHERE benchmark_name = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (benchmark_name, limit))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                result = dict(row)
                result['custom_metrics'] = json.loads(result['custom_metrics'] or '{}')
                result['metadata'] = json.loads(result['metadata'] or '{}')
                results.append(result)
            
            return results
    
    def get_performance_trends(self, benchmark_name: str, metric_name: str, 
                             days: int = 30) -> List[Tuple[datetime, float]]:
        """Get performance trends for a specific benchmark and metric."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT timestamp, metric_value 
                FROM performance_trends 
                WHERE benchmark_name = ? AND metric_name = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (benchmark_name, metric_name, since_date.isoformat()))
            
            rows = cursor.fetchall()
            trends = []
            
            for row in rows:
                timestamp = datetime.fromisoformat(row['timestamp'])
                trends.append((timestamp, row['metric_value']))
            
            return trends
    
    def get_suite_history(self, suite_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical suite execution results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM benchmark_suites 
                WHERE suite_name = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (suite_name, limit))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                result = dict(row)
                result['metadata'] = json.loads(result['metadata'] or '{}')
                results.append(result)
            
            return results
    
    def get_benchmark_statistics(self, benchmark_name: str, days: int = 30) -> Dict[str, Any]:
        """Get statistical summary for a benchmark."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_runs,
                    AVG(execution_time) as avg_execution_time,
                    MIN(execution_time) as min_execution_time,
                    MAX(execution_time) as max_execution_time,
                    AVG(memory_usage) as avg_memory_usage,
                    AVG(cpu_usage) as avg_cpu_usage,
                    AVG(throughput) as avg_throughput,
                    AVG(latency) as avg_latency
                FROM benchmark_results 
                WHERE benchmark_name = ? AND timestamp >= ?
            """, (benchmark_name, since_date.isoformat()))
            
            row = cursor.fetchone()
            
            if row and row['total_runs'] > 0:
                stats = dict(row)
                stats['success_rate'] = stats['successful_runs'] / stats['total_runs']
                return stats
            else:
                return {
                    'total_runs': 0,
                    'successful_runs': 0,
                    'success_rate': 0.0
                }
    
    def cleanup_old_results(self, cutoff_date: datetime) -> int:
        """Clean up old benchmark results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete old results
            cursor.execute("""
                DELETE FROM benchmark_results 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_results = cursor.rowcount
            
            # Delete old trends
            cursor.execute("""
                DELETE FROM performance_trends 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_trends = cursor.rowcount
            
            # Delete old suite records
            cursor.execute("""
                DELETE FROM benchmark_suites 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_suites = cursor.rowcount
            
            conn.commit()
            
            total_deleted = deleted_results + deleted_trends + deleted_suites
            logger.info(f"Cleaned up {total_deleted} old records (older than {days} days)")
            
            return total_deleted
    
    def get_recent_results(self, benchmark_name: str, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent results for a specific benchmark within the specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT * FROM benchmark_results 
                WHERE benchmark_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (benchmark_name, since_date.isoformat(), limit))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                result = dict(row)
                result['custom_metrics'] = json.loads(result['custom_metrics'] or '{}')
                result['metadata'] = json.loads(result['metadata'] or '{}')
                results.append(result)
            
            return results
    
    def get_all_recent_results(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all recent benchmark results within the specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT * FROM benchmark_results 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (since_date.isoformat(), limit))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                result = dict(row)
                result['custom_metrics'] = json.loads(result['custom_metrics'] or '{}')
                result['metadata'] = json.loads(result['metadata'] or '{}')
                results.append(result)
            
            return results
    
    def get_results_in_range(self, benchmark_name: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get benchmark results within a specific date range."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM benchmark_results 
                WHERE benchmark_name = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (benchmark_name, start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                result = dict(row)
                result['custom_metrics'] = json.loads(result['custom_metrics'] or '{}')
                result['metadata'] = json.loads(result['metadata'] or '{}')
                results.append(result)
            
            return results
    
    def export_results(self, output_file: str, benchmark_name: Optional[str] = None,
                      days: Optional[int] = None) -> None:
        """Export benchmark results to JSON file."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM benchmark_results"
            params = []
            
            conditions = []
            if benchmark_name:
                conditions.append("benchmark_name = ?")
                params.append(benchmark_name)
            
            if days:
                since_date = datetime.now() - timedelta(days=days)
                conditions.append("timestamp >= ?")
                params.append(since_date.isoformat())
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                result['custom_metrics'] = json.loads(result['custom_metrics'] or '{}')
                result['metadata'] = json.loads(result['metadata'] or '{}')
                results.append(result)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Exported {len(results)} results to {output_file}")