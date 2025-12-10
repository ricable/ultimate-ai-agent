# Troubleshooting Guide

This comprehensive guide covers common issues, debugging techniques, and error resolution strategies for SAFLA (Self-Aware Feedback Loop Algorithm). Our troubleshooting approach emphasizes systematic diagnosis and resolution.

## Table of Contents

1. [Troubleshooting Philosophy](#troubleshooting-philosophy)
2. [Common Issues](#common-issues)
3. [Debugging Techniques](#debugging-techniques)
4. [Error Categories](#error-categories)
5. [Component-Specific Issues](#component-specific-issues)
6. [Performance Issues](#performance-issues)
7. [Network and Connectivity](#network-and-connectivity)
8. [Database Issues](#database-issues)
9. [Memory and Resource Issues](#memory-and-resource-issues)
10. [Security Issues](#security-issues)
11. [Deployment Issues](#deployment-issues)
12. [Monitoring and Alerting](#monitoring-and-alerting)
13. [Recovery Procedures](#recovery-procedures)
14. [Prevention Strategies](#prevention-strategies)

## Troubleshooting Philosophy

SAFLA's troubleshooting approach follows systematic principles:

### Structured Diagnosis
- **Symptom Identification**: Clearly define the problem symptoms
- **Root Cause Analysis**: Trace issues to their underlying causes
- **Impact Assessment**: Understand the scope and severity of issues
- **Solution Prioritization**: Address critical issues first

### Evidence-Based Resolution
- **Log Analysis**: Use comprehensive logging for diagnosis
- **Metrics Review**: Analyze performance and health metrics
- **Test Validation**: Verify fixes with appropriate tests
- **Documentation**: Record solutions for future reference

### Proactive Prevention
- **Monitoring**: Continuous system health monitoring
- **Alerting**: Early warning systems for potential issues
- **Testing**: Comprehensive testing to prevent regressions
- **Automation**: Automated recovery for known issues

## Common Issues

### 1. Service Startup Failures

#### Symptoms
```bash
# Service fails to start
Error: Cannot start SAFLA service
Exit code: 1

# Port binding errors
Error: EADDRINUSE: address already in use :::3000

# Configuration errors
Error: Invalid configuration: missing required field 'database_url'
```

#### Diagnosis
```bash
# Check if port is in use
netstat -tulpn | grep :3000
lsof -i :3000

# Verify configuration
safla config validate
safla config show

# Check service logs
journalctl -u safla -f
docker logs safla-core
kubectl logs deployment/safla-core -n safla-production
```

#### Resolution
```bash
# Kill process using port
sudo kill -9 $(lsof -t -i:3000)

# Fix configuration
export DATABASE_URL="postgresql://user:pass@localhost:5432/safla"
safla config set database_url $DATABASE_URL

# Restart service
systemctl restart safla
docker-compose restart safla-core
kubectl rollout restart deployment/safla-core -n safla-production
```

### 2. Database Connection Issues

#### Symptoms
```bash
# Connection timeouts
Error: Connection timeout after 30000ms
SQLSTATE[HY000] [2002] Connection timed out

# Authentication failures
Error: FATAL: password authentication failed for user "safla"
Access denied for user 'safla'@'localhost'

# Database not found
Error: FATAL: database "safla_production" does not exist
```

#### Diagnosis
```bash
# Test database connectivity
pg_isready -h localhost -p 5432 -U safla
mysql -h localhost -u safla -p

# Check database status
systemctl status postgresql
systemctl status mysql

# Verify credentials
psql -h localhost -U safla -d safla_production
mysql -h localhost -u safla -p safla_production

# Check network connectivity
telnet db-host 5432
nc -zv db-host 5432
```

#### Resolution
```bash
# Fix connection string
export DATABASE_URL="postgresql://safla:correct_password@db-host:5432/safla_production"

# Create missing database
createdb -h localhost -U postgres safla_production
mysql -u root -p -e "CREATE DATABASE safla_production;"

# Reset password
sudo -u postgres psql -c "ALTER USER safla PASSWORD 'new_password';"
mysql -u root -p -e "SET PASSWORD FOR 'safla'@'localhost' = PASSWORD('new_password');"

# Update firewall rules
sudo ufw allow 5432
sudo firewall-cmd --add-port=5432/tcp --permanent
```

### 3. Vector Memory Issues

#### Symptoms
```bash
# Out of memory errors
Error: Cannot allocate memory for vector storage
MemoryError: Unable to allocate array with shape (100000, 768)

# Index corruption
Error: Vector index corrupted, rebuilding required
FAISS index file appears to be corrupted

# Similarity search failures
Error: Similarity search failed: index not trained
```

#### Diagnosis
```bash
# Check memory usage
free -h
ps aux | grep safla-vector
docker stats safla-vector-memory

# Verify vector storage
ls -la /app/data/vectors/
du -sh /app/data/vectors/

# Test vector operations
curl -X POST http://localhost:8000/vectors/test
safla vector health-check
```

#### Resolution
```bash
# Increase memory allocation
docker update --memory=4g safla-vector-memory
kubectl patch deployment safla-vector-memory -p '{"spec":{"template":{"spec":{"containers":[{"name":"safla-vector-memory","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# Rebuild vector index
safla vector rebuild-index
curl -X POST http://localhost:8000/vectors/rebuild

# Clear corrupted data
rm -rf /app/data/vectors/corrupted_index
safla vector initialize --force
```

### 4. MCP Communication Failures

#### Symptoms
```bash
# Connection refused
Error: Connection refused to MCP server at localhost:3001
ECONNREFUSED 127.0.0.1:3001

# Timeout errors
Error: MCP request timeout after 30000ms
Request to MCP server timed out

# Protocol errors
Error: Invalid MCP message format
MCP protocol version mismatch
```

#### Diagnosis
```bash
# Check MCP server status
curl http://localhost:3001/health
safla mcp status

# Test MCP connectivity
telnet localhost 3001
nc -zv mcp-server 3001

# Verify MCP configuration
safla mcp config show
cat /etc/safla/mcp-servers.json
```

#### Resolution
```bash
# Restart MCP server
systemctl restart safla-mcp
docker-compose restart safla-mcp-orchestrator

# Update MCP endpoints
safla mcp add-server --url http://new-mcp-server:3001
export MCP_SERVER_ENDPOINTS="http://mcp1:3001,http://mcp2:3001"

# Fix protocol version
safla mcp update-protocol --version 1.0
```

## Debugging Techniques

### 1. Log Analysis

#### Centralized Logging Setup
```yaml
# logging/fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/safla-*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name safla-logs
      type_name _doc
    </match>
```

#### Log Analysis Queries
```bash
# Search for errors in the last hour
grep -i error /var/log/safla/*.log | grep "$(date -d '1 hour ago' '+%Y-%m-%d %H')"

# Find performance issues
grep -E "(slow|timeout|latency)" /var/log/safla/performance.log

# Analyze error patterns
awk '/ERROR/ {print $0}' /var/log/safla/app.log | sort | uniq -c | sort -nr

# Search in Elasticsearch
curl -X GET "elasticsearch:9200/safla-logs/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {"range": {"@timestamp": {"gte": "now-1h"}}},
        {"match": {"level": "ERROR"}}
      ]
    }
  },
  "sort": [{"@timestamp": {"order": "desc"}}]
}'
```

### 2. Performance Profiling

#### CPU Profiling
```python
# profiling/cpu_profiler.py
import cProfile
import pstats
import io
from functools import wraps

class CPUProfiler:
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.is_profiling = False
    
    def start_profiling(self):
        """Start CPU profiling."""
        if not self.is_profiling:
            self.profiler.enable()
            self.is_profiling = True
    
    def stop_profiling(self):
        """Stop CPU profiling and return results."""
        if self.is_profiling:
            self.profiler.disable()
            self.is_profiling = False
            
            # Generate report
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            return s.getvalue()
        return None
    
    def profile_function(self, func):
        """Decorator to profile specific functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start_profiling()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profile_result = self.stop_profiling()
                if profile_result:
                    print(f"Profile for {func.__name__}:")
                    print(profile_result)
        return wrapper

# Usage
profiler = CPUProfiler()

@profiler.profile_function
def slow_function():
    # Function to profile
    pass
```

#### Memory Profiling
```python
# profiling/memory_profiler.py
import tracemalloc
import psutil
import gc
from typing import Dict, List

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []
        self.baseline = None
    
    def start_tracing(self):
        """Start memory tracing."""
        tracemalloc.start()
        self.baseline = tracemalloc.take_snapshot()
    
    def take_snapshot(self, name: str = None):
        """Take a memory snapshot."""
        if not tracemalloc.is_tracing():
            raise RuntimeError("Memory tracing not started")
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'name': name or f'snapshot_{len(self.snapshots)}',
            'snapshot': snapshot,
            'timestamp': time.time()
        })
        
        return snapshot
    
    def analyze_memory_growth(self) -> Dict:
        """Analyze memory growth since baseline."""
        if not self.baseline or not self.snapshots:
            return {}
        
        current = self.snapshots[-1]['snapshot']
        top_stats = current.compare_to(self.baseline, 'lineno')
        
        growth_analysis = {
            'total_growth': sum(stat.size_diff for stat in top_stats),
            'top_consumers': []
        }
        
        for stat in top_stats[:10]:  # Top 10
            growth_analysis['top_consumers'].append({
                'file': stat.traceback.format()[0] if stat.traceback.format() else 'unknown',
                'size_diff': stat.size_diff,
                'count_diff': stat.count_diff,
                'current_size': stat.size
            })
        
        return growth_analysis
    
    def get_system_memory_info(self) -> Dict:
        """Get system memory information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024,  # MB
            'gc_stats': {
                'generation_0': gc.get_count()[0],
                'generation_1': gc.get_count()[1],
                'generation_2': gc.get_count()[2]
            }
        }
```

### 3. Network Debugging

#### Connection Testing
```bash
# Test HTTP endpoints
curl -v http://localhost:3000/health
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:3000/api/status

# Test TCP connectivity
nc -zv hostname 3000
telnet hostname 3000

# Test DNS resolution
nslookup safla-service.default.svc.cluster.local
dig safla-service.default.svc.cluster.local

# Trace network path
traceroute safla-service
mtr safla-service

# Monitor network traffic
tcpdump -i any port 3000
netstat -tulpn | grep safla
ss -tulpn | grep safla
```

#### Network Performance Testing
```bash
# Bandwidth testing
iperf3 -s  # Server
iperf3 -c server-ip  # Client

# Latency testing
ping -c 10 safla-service
hping3 -S -p 3000 -c 10 safla-service

# Load testing
ab -n 1000 -c 10 http://localhost:3000/health
wrk -t12 -c400 -d30s http://localhost:3000/api/test
```

## Error Categories

### 1. Startup Errors

#### Configuration Errors
```python
# config/validator.py
class ConfigurationError(Exception):
    """Configuration validation error."""
    pass

def validate_configuration(config: Dict) -> List[str]:
    """Validate SAFLA configuration."""
    errors = []
    
    # Required fields
    required_fields = [
        'database_url',
        'redis_url',
        'vector_dimension',
        'mcp_servers'
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate database URL
    if 'database_url' in config:
        if not config['database_url'].startswith(('postgresql://', 'mysql://')):
            errors.append("Invalid database URL format")
    
    # Validate vector dimension
    if 'vector_dimension' in config:
        if not isinstance(config['vector_dimension'], int) or config['vector_dimension'] <= 0:
            errors.append("Vector dimension must be a positive integer")
    
    # Validate MCP servers
    if 'mcp_servers' in config:
        if not isinstance(config['mcp_servers'], list) or len(config['mcp_servers']) == 0:
            errors.append("MCP servers must be a non-empty list")
    
    return errors

# Usage
try:
    config = load_configuration()
    errors = validate_configuration(config)
    if errors:
        raise ConfigurationError(f"Configuration errors: {', '.join(errors)}")
except ConfigurationError as e:
    logger.error(f"Configuration validation failed: {e}")
    sys.exit(1)
```

#### Dependency Errors
```python
# dependencies/checker.py
import importlib
import subprocess
import sys

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'numpy',
        'faiss-cpu',
        'redis',
        'psycopg2',
        'asyncio',
        'aiohttp'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_external_services():
    """Check if external services are available."""
    services = [
        ('Redis', 'redis-cli ping'),
        ('PostgreSQL', 'pg_isready'),
        ('Vector Store', 'curl -f http://localhost:8000/health')
    ]
    
    failed_services = []
    
    for service_name, check_command in services:
        try:
            result = subprocess.run(
                check_command.split(),
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                failed_services.append(service_name)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            failed_services.append(service_name)
    
    if failed_services:
        print(f"Failed to connect to services: {', '.join(failed_services)}")
        return False
    
    return True
```

### 2. Runtime Errors

#### Memory Errors
```python
# errors/memory_handler.py
import psutil
import gc
import logging

logger = logging.getLogger(__name__)

class MemoryError(Exception):
    """Memory-related error."""
    pass

class MemoryMonitor:
    def __init__(self, threshold_percent=85, critical_percent=95):
        self.threshold_percent = threshold_percent
        self.critical_percent = critical_percent
        self.process = psutil.Process()
    
    def check_memory_usage(self):
        """Check current memory usage."""
        memory_percent = self.process.memory_percent()
        
        if memory_percent > self.critical_percent:
            # Critical memory usage - force garbage collection
            logger.warning(f"Critical memory usage: {memory_percent:.1f}%")
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Check again after GC
            new_memory_percent = self.process.memory_percent()
            if new_memory_percent > self.critical_percent:
                raise MemoryError(f"Critical memory usage: {new_memory_percent:.1f}%")
        
        elif memory_percent > self.threshold_percent:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
            # Trigger gentle cleanup
            gc.collect()
        
        return memory_percent
    
    def get_memory_info(self):
        """Get detailed memory information."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_rss': memory_info.rss / 1024 / 1024,  # MB
            'process_vms': memory_info.vms / 1024 / 1024,  # MB
            'process_percent': self.process.memory_percent(),
            'system_total': system_memory.total / 1024 / 1024,  # MB
            'system_available': system_memory.available / 1024 / 1024,  # MB
            'system_percent': system_memory.percent
        }
```

#### Performance Errors
```python
# errors/performance_handler.py
import time
import statistics
from collections import deque
from typing import Dict, List

class PerformanceError(Exception):
    """Performance-related error."""
    pass

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.error_counts = deque(maxlen=window_size)
        self.start_time = time.time()
    
    def record_request(self, response_time: float, success: bool = True):
        """Record request performance."""
        self.response_times.append(response_time)
        self.error_counts.append(0 if success else 1)
        
        # Check for performance issues
        self._check_performance_thresholds()
    
    def _check_performance_thresholds(self):
        """Check if performance thresholds are exceeded."""
        if len(self.response_times) < 10:
            return  # Not enough data
        
        # Calculate metrics
        avg_response_time = statistics.mean(self.response_times)
        p95_response_time = statistics.quantiles(self.response_times, n=20)[18]
        error_rate = sum(self.error_counts) / len(self.error_counts)
        
        # Check thresholds
        if avg_response_time > 5.0:  # 5 seconds
            raise PerformanceError(f"High average response time: {avg_response_time:.2f}s")
        
        if p95_response_time > 10.0:  # 10 seconds
            raise PerformanceError(f"High P95 response time: {p95_response_time:.2f}s")
        
        if error_rate > 0.1:  # 10% error rate
            raise PerformanceError(f"High error rate: {error_rate:.1%}")
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        if not self.response_times:
            return {}
        
        return {
            'avg_response_time': statistics.mean(self.response_times),
            'p95_response_time': statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) > 20 else max(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'error_rate': sum(self.error_counts) / len(self.error_counts),
            'total_requests': len(self.response_times),
            'uptime': time.time() - self.start_time
        }
```

## Component-Specific Issues

### 1. Vector Memory Manager Issues

#### Index Corruption
```python
# vector/recovery.py
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorIndexRecovery:
    def __init__(self, data_dir: str, backup_dir: str):
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def detect_corruption(self) -> bool:
        """Detect if vector index is corrupted."""
        try:
            # Try to load the index
            import faiss
            index_file = self.data_dir / "vector_index.faiss"
            
            if not index_file.exists():
                logger.warning("Vector index file not found")
                return True
            
            # Attempt to load index
            index = faiss.read_index(str(index_file))
            
            # Basic validation
            if index.ntotal == 0:
                logger.warning("Vector index is empty")
                return True
            
            # Test search functionality
            test_vector = np.random.random((1, index.d)).astype('float32')
            distances, indices = index.search(test_vector, min(10, index.ntotal))
            
            return False
            
        except Exception as e:
            logger.error(f"Vector index corruption detected: {e}")
            return True
    
    def backup_index(self) -> str:
        """Create backup of current index."""
        timestamp = int(time.time())
        backup_path = self.backup_dir / f"vector_index_backup_{timestamp}"
        
        try:
            shutil.copytree(self.data_dir, backup_path)
            logger.info(f"Vector index backed up to {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to backup vector index: {e}")
            raise
    
    def restore_from_backup(self, backup_path: str = None) -> bool:
        """Restore index from backup."""
        if backup_path is None:
            # Find latest backup
            backups = list(self.backup_dir.glob("vector_index_backup_*"))
            if not backups:
                logger.error("No backups found")
                return False
            backup_path = max(backups, key=lambda p: p.stat().st_mtime)
        
        try:
            # Remove corrupted index
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            
            # Restore from backup
            shutil.copytree(backup_path, self.data_dir)
            logger.info(f"Vector index restored from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore vector index: {e}")
            return False
    
    def rebuild_index(self, vectors_source: str = None) -> bool:
        """Rebuild index from source data."""
        try:
            # Initialize new index
            from src.vector_memory.manager import VectorMemoryManager
            
            # Remove old index
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            self.data_dir.mkdir(exist_ok=True)
            
            # Create new manager
            manager = VectorMemoryManager(
                embedding_dim=768,  # Default dimension
                similarity_metric="cosine",
                storage_path=str(self.data_dir)
            )
            
            # Rebuild from source if available
            if vectors_source and os.path.exists(vectors_source):
                logger.info(f"Rebuilding index from {vectors_source}")
                # Load and re-index vectors
                # Implementation depends on source format
            
            logger.info("Vector index rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild vector index: {e}")
            return False
```

#### Memory Optimization
```python
# vector/optimization.py
import gc
import psutil
from typing import Optional

class VectorMemoryOptimizer:
    def __init__(self, max_memory_percent: float = 80):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
    
    def optimize_memory_usage(self):
        """Optimize vector memory usage."""
        current_memory = self.process.memory_percent()
        
        if current_memory > self.max_memory_percent:
            logger.warning(f"High memory usage: {current_memory:.1f}%")
            
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Check memory again
            new_memory = self.process.memory_percent()
            
            if new_memory > self.max_memory_percent:
                # More aggressive optimization needed
                self._aggressive_memory_cleanup()
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup."""
        # Clear caches
        if hasattr(self, 'vector_cache'):
            self.vector_cache.clear()
        
        # Compact vector storage
        self._compact_vector_storage()
        
        # Force full garbage collection
        for generation in range(3):
            gc.collect(generation)
    
    def _compact_vector_storage(self):
        """Compact vector storage to reduce memory usage."""
        # Implementation depends on storage backend
        pass
```

### 2. MCP Orchestrator Issues

#### Connection Pool Management
```python
# mcp/connection_recovery.py
import asyncio
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MCPConnectionRecovery:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.failed_connections = {}
        self.recovery_tasks = {}
    
    async def recover_connection(self, server_id: str, server_config: Dict):
        """Recover failed MCP connection."""
        if server_id in self.recovery_tasks:
            # Recovery already in progress
            return await self.recovery_tasks[server_id]
        
        # Start recovery task
        self.recovery_tasks[server_id] = asyncio.create_task(
            self._recovery_loop(server_id, server_config)
        )
        
        try:
            return await self.recovery_tasks[server_id]
        finally:
            self.recovery_tasks.pop(server_id, None)
    
    async def _recovery_loop(self, server_id: str, server_config: Dict):
        """Recovery loop for MCP connection."""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                logger.info(f"Attempting to recover MCP connection {server_id} (attempt {retry_count + 1})")
                
                # Attempt connection
                connection = await self._create_connection(server_config)
                
                # Test connection
                await self._test_connection(connection)
                
                logger.info(f"Successfully recovered MCP connection {server_id}")
                self.failed_connections.pop(server_id, None)
                return connection
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"MCP connection recovery failed for {server_id}: {e}")
                
                if retry_count < self.max_retries:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    await asyncio.sleep(delay)
        
        # All retries failed
        logger.error(f"Failed to recover MCP connection {server_id} after {self.max_retries} attempts")
        self.failed_connections[server_id] = time.time()
        raise ConnectionError(f"Cannot recover MCP connection {server_id}")
    
    async def _create_connection(self, server_config: Dict):
        """Create new MCP connection."""
        # Implementation depends on MCP protocol
        pass
    
    async def _test_connection(self, connection):
        """Test MCP connection health."""
        # Send ping or health check
        pass
```

#### Task Queue Recovery
```python
# mcp/task_recovery.py
import asyncio
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class MCPTaskRecovery:
    def __init__(self, persistence_file: str = "mcp_tasks.json"):
        self.persistence_file = persistence_file
        self.failed_tasks = []
        self.retry_queue = asyncio.Queue()
    
    async def save_failed_task(self, task: Dict[str, Any]):
        """Save failed task for retry."""
        task['failed_at'] = time.time()
        task['retry_count'] = task.get('retry_count', 0) + 1
        
        self.failed_tasks.append(task)
        await self.retry_queue.put(task)
        
        # Persist to disk
        await self._persist_tasks()
    
    async def load_failed_tasks(self):
        """Load failed tasks from persistence."""
        try:
            with open(self.persistence_file, 'r') as f:
                tasks = json.load(f)
            
            for task in tasks:
                await self.retry_queue.put(task)
            
            logger.info(f"Loaded {len(tasks)} failed tasks for retry")
            
        except FileNotFoundError:
            logger.info("No failed tasks file found")
        except Exception as e:
            logger.error(f"Failed to load failed tasks: {e}")
    
    async def retry_failed_tasks(self):
        """Retry failed tasks."""
        while True:
            try:
                # Get task from retry queue
                task = await asyncio.wait_for(self.retry_queue.get(), timeout=1.0)
                
                # Check if task should be retried
                if self._should_retry_task(task):
                    await self._retry_task(task)
                else:
                    logger.warning(f"Task {task.get('id')} exceeded max retries")
                
            except asyncio.TimeoutError:
                # No tasks to retry
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in task retry loop: {e}")
                await asyncio.sleep(1)
    
    def _should_retry_task(self, task: Dict[str, Any]) -> bool:
        """Check if task should be retried."""
        max_retries = task.get('max_retries', 3)
        retry_count = task.get('retry_count', 0)
        
        return retry_count <= max_retries
    
    async def _retry_task(self, task: Dict[str, Any]):
        """Retry a failed task."""
        try:
            logger.info(f"Retrying task {task.get('id')} (attempt {task.get('retry_count')})")
            
            # Execute task
            result = await self._execute_task(task)
            
            # Remove from failed tasks on success
            self.failed_tasks = [t for t in self.failed_tasks if t.get('id') != task.get('id')]
            await self._persist_tasks()
            
            logger.info(f"Task {task.get('id')} retry successful")
            
        except Exception as e:
            logger.error(f"Task {task.get('id')} retry failed: {e}")
            await self.save_failed_task(task)
    
    async def _execute_task(self, task: Dict[str, Any]):
        """Execute MCP task."""
        # Implementation depends on task type
        pass
    
    async def _persist_tasks(self):
        """Persist failed tasks to disk."""
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(self.failed_tasks, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist tasks: {e}")
```

## Performance Issues

### 1. Slow Response Times

#### Diagnosis
```python
# performance/diagnosis.py
import time
import asyncio
import statistics
from typing import Dict, List, Callable

class PerformanceDiagnostic:
    def __init__(self):
        self.measurements = {}
        self.bottlenecks = []
    
    async def diagnose_slow_response(self, endpoint: str) -> Dict:
        """Diagnose slow response times for an endpoint."""
        measurements = {
            'database_query_time': await self._measure_database_performance(),
            'vector_search_time': await self._measure_vector_search_performance(),
            'mcp_communication_time': await self._measure_mcp_performance(),
            'memory_usage': self._measure_memory_usage(),
            'cpu_usage': self._measure_cpu_usage()
        }
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(measurements)
        
        return {
            'endpoint': endpoint,
            'measurements': measurements,
            'bottlenecks': bottlenecks,
            'recommendations': self._generate_recommendations(bottlenecks)
        }
    
    async def _measure_database_performance(self) -> float:
        """Measure database query performance."""
        start_time = time.time()
        
        try:
            # Execute test query
            await self._execute_test_query()
            return time.time() - start_time
        except Exception as e:
            logger.error(f"Database performance test failed: {e}")
            return float('inf')
    
    async def _measure_vector_search_performance(self) -> float:
        """Measure vector search performance."""
        start_time = time.time()
        
        try:
            # Execute test vector search
            await self._execute_test_vector_search()
            return time.time() - start_time
        except Exception as e:
            logger.error(f"Vector search performance test failed: {e}")
            return float('inf')
    
    async def _measure_mcp_performance(self) -> float:
        """Measure MCP communication performance."""
        start_time = time.time()
        
        try:
            # Execute test MCP request
            await self._execute_test_mcp_request()
            return time.time() - start_time
        except Exception as e:
            logger.error(f"MCP performance test failed: {e}")
            return float('inf')
    
    def _identify_bottlenecks(self, measurements: Dict) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Define thresholds
        thresholds = {
            'database_query_time': 0.1,  # 100ms
            'vector_search_time': 0.05,  # 50ms
            'mcp_communication_time': 0.02,  # 20ms
            'memory_usage': 80,  # 80%
            'cpu_usage': 80  # 80%
        }
        
        for metric, value in measurements.items():
            if metric in thresholds and value > thresholds[metric]:
                bottlenecks.append(metric)
        
        return bottlenecks
    
    def _generate_recommendations(self, bottlenecks: List[str]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        recommendation_map = {
            'database_query_time': [
                "Add database indexes for frequently queried columns",
                "Optimize database queries",
                "Consider database connection pooling",
                "Review database configuration"
            ],
            'vector_search_time': [
                "Optimize vector index configuration",
                "Consider using GPU acceleration",
                "Implement vector caching",
                "Reduce vector dimensionality if possible"
            ],
            'mcp_communication_time': [
                "Implement connection pooling for MCP servers",
                "Add request batching",
                "Optimize network configuration",
                "Consider local MCP server deployment"
            ],
            'memory_usage': [
                "Implement memory caching strategies",
                "Optimize data structures",
                "Add garbage collection tuning",
                "Consider increasing available memory"
            ],
            'cpu_usage': [
                "Optimize CPU-intensive algorithms",
                "Implement parallel processing",
                "Consider horizontal scaling",
                "Profile and optimize hot code paths"
            ]
        }
        
        for bottleneck in bottlenecks:
            if bottleneck in recommendation_map:
                recommendations.extend(recommendation_map[bottleneck])
        
        return recommendations
```

### 2. Memory Leaks

#### Detection
```python
# performance/memory_leak_detector.py
import gc
import tracemalloc
import psutil
import time
from typing import Dict, List, Optional

class MemoryLeakDetector:
    def __init__(self, check_interval: int = 300):  # 5 minutes
        self.check_interval = check_interval
        self.baseline_memory = None
        self.memory_history = []
        self.leak_threshold = 50  # MB growth over time
        self.monitoring = False
    
    def start_monitoring(self):
        """Start memory leak monitoring."""
        if not self.monitoring:
            tracemalloc.start()
            self.baseline_memory = self._get_memory_usage()
            self.monitoring = True
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self):
        """Stop memory leak monitoring."""
        if self.monitoring:
            tracemalloc.stop()
            self.monitoring = False
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                current_memory = self._get_memory_usage()
                self.memory_history.append({
                    'timestamp': time.time(),
                    'memory_usage': current_memory
                })
                
                # Keep only recent history (last 24 hours)
                cutoff_time = time.time() - 86400
                self.memory_history = [
                    entry for entry in self.memory_history
                    if entry['timestamp'] > cutoff_time
                ]
                
                # Check for memory leaks
                leak_detected = self._detect_memory_leak()
                if leak_detected:
                    await self._handle_memory_leak()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _detect_memory_leak(self) -> bool:
        """Detect if there's a memory leak."""
        if len(self.memory_history) < 10:
            return False
        
        # Calculate memory growth trend
        recent_entries = self.memory_history[-10:]
        oldest_entry = recent_entries[0]
        newest_entry = recent_entries[-1]
        
        memory_growth = newest_entry['memory_usage'] - oldest_entry['memory_usage']
        time_diff = newest_entry['timestamp'] - oldest_entry['timestamp']
        
        # Check if memory is consistently growing
        if memory_growth > self.leak_threshold and time_diff > 3600:  # 1 hour
            return True
        
        return False
    
    async def _handle_memory_leak(self):
        """Handle detected memory leak."""
        logger.warning("Memory leak detected, performing analysis")
        
        # Take memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Log top memory consumers
        logger.warning("Top memory consumers:")
        for index, stat in enumerate(top_stats[:10], 1):
            logger.warning(f"{index}. {stat}")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Forced garbage collection, freed {collected} objects")
        
        # Check if leak persists
        new_memory = self._get_memory_usage()
        if new_memory > self.baseline_memory + self.leak_threshold:
            logger.error("Memory leak persists after garbage collection")
            # Could trigger alerts or restart procedures here
```

This troubleshooting guide provides comprehensive coverage of common issues and systematic approaches to diagnosing and resolving problems in SAFLA deployments.