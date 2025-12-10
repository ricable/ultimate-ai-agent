# MCP Optimizer Mode Examples

## Before/After Performance Comparisons

### API Response Time Optimization
**Before:**
- Average response time: 2.8 seconds
- 95th percentile: 5.2 seconds
- Throughput: 45 requests/second
- Memory usage: 512MB peak

**After:**
- Average response time: 1.5 seconds (46% improvement)
- 95th percentile: 2.1 seconds (60% improvement)
- Throughput: 120 requests/second (167% improvement)
- Memory usage: 320MB peak (37% reduction)

### Database Query Optimization
**Before:**
```sql
-- Slow query taking 15.2 seconds
SELECT o.*, c.name, p.title
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id
WHERE o.created_at > '2024-01-01';
```

**After:**
```sql
-- Optimized query taking 0.8 seconds (95% improvement)
SELECT o.id, o.total, c.name, p.title
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id
WHERE o.created_at > '2024-01-01'
AND o.status IN ('completed', 'shipped')
ORDER BY o.created_at DESC
LIMIT 1000;

-- Added indexes:
CREATE INDEX idx_orders_created_status ON orders(created_at, status);
CREATE INDEX idx_orders_customer ON orders(customer_id);
```

### Memory Usage Optimization
**Before:**
- Peak memory: 2.1GB
- Memory leaks: 45MB/hour
- GC pressure: High (15% CPU)

**After:**
- Peak memory: 890MB (58% reduction)
- Memory leaks: 0MB/hour (eliminated)
- GC pressure: Low (3% CPU, 80% reduction)

## Real-World Optimization Scenarios

### 1. Web API Performance Optimization

#### Scenario: E-commerce API Response Time Optimization
```typescript
// Initial analysis using MCP tools
await use_mcp_tool({
  server_name: "safla",
  tool_name: "analyze_performance_bottlenecks",
  arguments: {
    duration_seconds: 600,
    include_memory_profile: true
  }
});

// Baseline measurement
const baseline = await use_mcp_tool({
  server_name: "safla",
  tool_name: "benchmark_mcp_throughput",
  arguments: {
    request_count: 1000,
    concurrent_connections: 50,
    payload_size: "medium"
  }
});

// Optimization implementation
const optimizations = [
  {
    name: "Database Query Optimization",
    implementation: async () => {
      // Add database indexes
      // Optimize N+1 queries
      // Implement query caching
    },
    expectedImprovement: 40
  },
  {
    name: "Response Compression",
    implementation: async () => {
      // Enable gzip compression
      // Implement response caching
    },
    expectedImprovement: 25
  },
  {
    name: "Connection Pooling",
    implementation: async () => {
      // Configure database connection pools
      // Optimize connection lifecycle
    },
    expectedImprovement: 15
  }
];
```

**Results:**
- Response time: 45% improvement (800ms → 440ms)
- Throughput: 120% improvement (500 RPS → 1100 RPS)
- Memory usage: 30% reduction
- Error rate: 80% reduction

### 2. Database Query Optimization

#### Scenario: Large Dataset Analytics Query Optimization
```sql
-- Before optimization
SELECT 
    u.id, u.name, u.email,
    COUNT(o.id) as order_count,
    SUM(o.total) as total_spent,
    AVG(o.total) as avg_order_value
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
GROUP BY u.id, u.name, u.email
ORDER BY total_spent DESC;

-- Execution time: 15.2 seconds
-- Rows examined: 2,500,000

-- After optimization with indexes
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at);
CREATE INDEX CONCURRENTLY idx_orders_user_id_total ON orders(user_id, total);

-- Optimized query with materialized view
CREATE MATERIALIZED VIEW user_order_stats AS
SELECT 
    u.id, u.name, u.email,
    COALESCE(stats.order_count, 0) as order_count,
    COALESCE(stats.total_spent, 0) as total_spent,
    COALESCE(stats.avg_order_value, 0) as avg_order_value
FROM users u
LEFT JOIN (
    SELECT 
        user_id,
        COUNT(*) as order_count,
        SUM(total) as total_spent,
        AVG(total) as avg_order_value
    FROM orders
    GROUP BY user_id
) stats ON u.id = stats.user_id
WHERE u.created_at >= '2023-01-01';

-- Execution time: 0.8 seconds (95% improvement)
-- Rows examined: 150,000
```

**MCP Integration:**
```typescript
// Monitor query performance
await use_mcp_tool({
  server_name: "safla",
  tool_name: "benchmark_memory_performance",
  arguments: {
    test_duration: 300,
    memory_patterns: ["database_query"]
  }
});
```

### 3. Memory Usage Optimization

#### Scenario: Python Data Processing Pipeline
```python
# Before optimization - Memory inefficient
def process_large_dataset_inefficient(file_path):
    # Load entire dataset into memory
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Process all data at once
    results = []
    for item in data:
        processed = expensive_computation(item)
        results.append(processed)
    
    return results

# After optimization - Memory efficient
def process_large_dataset_optimized(file_path, batch_size=1000):
    def batch_generator():
        batch = []
        with open(file_path, 'r') as f:
            for line in f:
                batch.append(json.loads(line))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
    
    # Process in batches with generator
    for batch in batch_generator():
        # Use multiprocessing for CPU-intensive tasks
        with multiprocessing.Pool() as pool:
            batch_results = pool.map(expensive_computation, batch)
            yield from batch_results

# Memory usage: 95% reduction (8GB → 400MB)
# Processing time: 60% improvement due to parallelization
```

**MCP Monitoring:**
```typescript
// Monitor memory optimization
await use_mcp_tool({
  server_name: "safla",
  tool_name: "optimize_memory_usage",
  arguments: {
    optimization_level: "balanced",
    target_memory_mb: 512
  }
});

// Validate memory operations
await use_mcp_tool({
  server_name: "safla",
  tool_name: "validate_memory_operations",
  arguments: {
    test_data_size: 50,
    include_stress_test: true
  }
});
```

### 4. Frontend Performance Optimization

#### Scenario: React Application Bundle Size and Load Time
```typescript
// Before optimization - Large bundle
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Analytics from './components/Analytics';
import Reports from './components/Reports';
import Settings from './components/Settings';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </BrowserRouter>
  );
}

// After optimization - Code splitting and lazy loading
import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LoadingSpinner from './components/LoadingSpinner';

// Lazy load components
const Dashboard = React.lazy(() => import('./components/Dashboard'));
const Analytics = React.lazy(() => import('./components/Analytics'));
const Reports = React.lazy(() => import('./components/Reports'));
const Settings = React.lazy(() => import('./components/Settings'));

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

// Webpack configuration for optimization
module.exports = {
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all',
        },
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
};
```

**Results:**
- Initial bundle size: 65% reduction (2.1MB → 740KB)
- First contentful paint: 40% improvement (3.2s → 1.9s)
- Time to interactive: 50% improvement (5.1s → 2.5s)

### 5. Microservices Communication Optimization

#### Scenario: gRPC Service Performance Tuning
```typescript
// Before optimization - Basic gRPC client
import { UserServiceClient } from './generated/user_service_grpc_pb';
import { GetUserRequest } from './generated/user_service_pb';

class UserService {
  private client: UserServiceClient;

  constructor() {
    this.client = new UserServiceClient('localhost:50051');
  }

  async getUser(userId: string) {
    const request = new GetUserRequest();
    request.setUserId(userId);
    
    return new Promise((resolve, reject) => {
      this.client.getUser(request, (error, response) => {
        if (error) reject(error);
        else resolve(response);
      });
    });
  }
}

// After optimization - Connection pooling and batching
import { UserServiceClient } from './generated/user_service_grpc_pb';
import { GetUserRequest, GetUsersRequest } from './generated/user_service_pb';
import { ChannelCredentials, ChannelOptions } from '@grpc/grpc-js';

class OptimizedUserService {
  private client: UserServiceClient;
  private batchQueue: string[] = [];
  private batchTimeout: NodeJS.Timeout | null = null;

  constructor() {
    const options: ChannelOptions = {
      'grpc.keepalive_time_ms': 30000,
      'grpc.keepalive_timeout_ms': 5000,
      'grpc.keepalive_permit_without_calls': true,
      'grpc.http2.max_pings_without_data': 0,
      'grpc.http2.min_time_between_pings_ms': 10000,
      'grpc.http2.min_ping_interval_without_data_ms': 300000,
    };

    this.client = new UserServiceClient(
      'localhost:50051',
      ChannelCredentials.createInsecure(),
      options
    );
  }

  async getUser(userId: string): Promise<any> {
    return new Promise((resolve, reject) => {
      this.batchQueue.push(userId);
      
      if (this.batchTimeout) {
        clearTimeout(this.batchTimeout);
      }
      
      this.batchTimeout = setTimeout(() => {
        this.processBatch().then(results => {
          const userIndex = this.batchQueue.indexOf(userId);
          resolve(results[userIndex]);
        }).catch(reject);
      }, 10); // 10ms batch window
    });
  }

  private async processBatch(): Promise<any[]> {
    const userIds = [...this.batchQueue];
    this.batchQueue = [];
    
    const request = new GetUsersRequest();
    request.setUserIdsList(userIds);
    
    return new Promise((resolve, reject) => {
      this.client.getUsers(request, (error, response) => {
        if (error) reject(error);
        else resolve(response.getUsersList());
      });
    });
  }
}
```

**Performance Improvements:**
- Request latency: 70% reduction (50ms → 15ms)
- Throughput: 300% improvement (200 RPS → 800 RPS)
- Connection overhead: 85% reduction
- Resource usage: 40% reduction

### 6. Machine Learning Model Optimization

#### Scenario: TensorFlow Model Inference Optimization
```python
# Before optimization - Standard model loading
import tensorflow as tf
import numpy as np

class StandardModelInference:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def predict(self, input_data):
        # Single prediction
        return self.model.predict(np.expand_dims(input_data, axis=0))

# After optimization - Quantized model with batching
import tensorflow as tf
import numpy as np
from collections import deque
import threading
import time

class OptimizedModelInference:
    def __init__(self, model_path, batch_size=32, max_wait_time=0.1):
        # Load quantized model
        self.interpreter = tf.lite.Interpreter(model_path=f"{model_path}.tflite")
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batch_queue = deque()
        self.result_futures = {}
        
        # Start batch processing thread
        self.processing_thread = threading.Thread(target=self._process_batches)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def predict(self, input_data):
        future = threading.Event()
        request_id = id(future)
        
        self.batch_queue.append((request_id, input_data, future))
        future.wait()
        
        result = self.result_futures.pop(request_id)
        return result
    
    def _process_batches(self):
        while True:
            if len(self.batch_queue) >= self.batch_size:
                self._process_current_batch()
            elif self.batch_queue:
                time.sleep(self.max_wait_time)
                if self.batch_queue:
                    self._process_current_batch()
            else:
                time.sleep(0.01)
    
    def _process_current_batch(self):
        batch_items = []
        batch_data = []
        
        # Collect batch
        while self.batch_queue and len(batch_items) < self.batch_size:
            item = self.batch_queue.popleft()
            batch_items.append(item)
            batch_data.append(item[1])
        
        if not batch_items:
            return
        
        # Process batch
        batch_input = np.array(batch_data)
        self.interpreter.set_tensor(self.input_details[0]['index'], batch_input)
        self.interpreter.invoke()
        batch_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Return results
        for i, (request_id, _, future) in enumerate(batch_items):
            self.result_futures[request_id] = batch_output[i]
            future.set()

# Model quantization script
def quantize_model(model_path, output_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    quantized_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(quantized_model)
```

**Performance Improvements:**
- Model size: 75% reduction (400MB → 100MB)
- Inference time: 60% improvement (100ms → 40ms)
- Memory usage: 50% reduction (2GB → 1GB)
- Throughput: 400% improvement (10 RPS → 50 RPS)

## Benchmark Templates

### 1. API Performance Benchmark
```typescript
interface APIBenchmarkConfig {
  baseUrl: string;
  endpoints: string[];
  concurrency: number;
  duration: number;
  rampUp: number;
}

class APIPerformanceBenchmark {
  async runBenchmark(config: APIBenchmarkConfig): Promise<BenchmarkResults> {
    const results = await use_mcp_tool({
      server_name: "safla",
      tool_name: "benchmark_mcp_throughput",
      arguments: {
        request_count: config.duration * config.concurrency,
        concurrent_connections: config.concurrency,
        payload_size: "medium"
      }
    });
    
    return {
      averageResponseTime: results.average_response_time,
      throughput: results.requests_per_second,
      errorRate: results.error_rate,
      p95ResponseTime: results.p95_response_time,
      p99ResponseTime: results.p99_response_time
    };
  }
}
```

### 2. Database Performance Benchmark
```sql
-- Database benchmark queries
-- Test 1: Simple SELECT performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM users WHERE email = 'test@example.com';

-- Test 2: Complex JOIN performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY u.id, u.name
ORDER BY total_spent DESC
LIMIT 100;

-- Test 3: Aggregation performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    DATE_TRUNC('day', created_at) as day,
    COUNT(*) as order_count,
    AVG(total) as avg_order_value
FROM orders
WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY day;
```

### 3. Memory Performance Benchmark
```python
import psutil
import time
import gc
from typing import List, Dict, Any

class MemoryBenchmark:
    def __init__(self):
        self.baseline_memory = psutil.Process().memory_info().rss
    
    def measure_memory_usage(self, func, *args, **kwargs):
        # Force garbage collection
        gc.collect()
        
        # Measure before
        before_memory = psutil.Process().memory_info().rss
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure after
        end_time = time.time()
        after_memory = psutil.Process().memory_info().rss
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_used': after_memory - before_memory,
            'peak_memory': after_memory,
            'memory_efficiency': (after_memory - before_memory) / (end_time - start_time)
        }
    
    def run_memory_stress_test(self, iterations: int = 1000):
        memory_usage = []
        
        for i in range(iterations):
            # Allocate memory
            data = [i] * 10000
            current_memory = psutil.Process().memory_info().rss
            memory_usage.append(current_memory)
            
            # Cleanup
            del data
            if i % 100 == 0:
                gc.collect()
        
        return {
            'max_memory': max(memory_usage),
            'avg_memory': sum(memory_usage) / len(memory_usage),
            'memory_growth': memory_usage[-1] - memory_usage[0]
        }
```

## Optimization Checklists

### 1. Code Optimization Checklist
- [ ] Profile application to identify bottlenecks
- [ ] Analyze algorithm complexity (time and space)
- [ ] Optimize data structures for access patterns
- [ ] Implement caching for expensive operations
- [ ] Minimize object creation in hot paths
- [ ] Use appropriate data types and collections
- [ ] Optimize loops and iterations
- [ ] Implement lazy loading where appropriate
- [ ] Use connection pooling for external services
- [ ] Enable compression for data transfer

### 2. Database Optimization Checklist
- [ ] Analyze query execution plans
- [ ] Create appropriate indexes
- [ ] Optimize JOIN operations
- [ ] Implement query result caching
- [ ] Use prepared statements
- [ ] Optimize database schema design
- [ ] Implement connection pooling
- [ ] Monitor and tune database parameters
- [ ] Use read replicas for read-heavy workloads
- [ ] Implement database partitioning if needed

### 3. Frontend Optimization Checklist
- [ ] Implement code splitting and lazy loading
- [ ] Optimize bundle size with tree shaking
- [ ] Compress and optimize images
- [ ] Use CDN for static assets
- [ ] Implement service workers for caching
- [ ] Minimize and compress CSS/JS
- [ ] Use efficient CSS selectors
- [ ] Optimize rendering performance
- [ ] Implement virtual scrolling for large lists
- [ ] Use web workers for heavy computations

### 4. Infrastructure Optimization Checklist
- [ ] Configure appropriate server resources
- [ ] Implement load balancing
- [ ] Use content delivery networks (CDN)
- [ ] Configure caching layers (Redis, Memcached)
- [ ] Optimize network configuration
- [ ] Implement monitoring and alerting
- [ ] Use container orchestration efficiently
- [ ] Configure auto-scaling policies
- [ ] Optimize storage performance
- [ ] Implement backup and disaster recovery

## Monitoring and Alerting Templates

### 1. Performance Monitoring Dashboard
```typescript
interface PerformanceMetrics {
  responseTime: {
    avg: number;
    p95: number;
    p99: number;
  };
  throughput: {
    requestsPerSecond: number;
    errorsPerSecond: number;
  };
  resources: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
  };
}

class PerformanceMonitor {
  async collectMetrics(): Promise<PerformanceMetrics> {
    return await use_mcp_tool({
      server_name: "safla",
      tool_name: "monitor_system_health",
      arguments: {
        check_interval: 60,
        alert_thresholds: {
          cpu_usage: 80,
          memory_usage: 85,
          response_time_p95: 1000
        }
      }
    });
  }
  
  async setupAlerts() {
    // Configure performance alerts
    const alertRules = [
      {
        metric: 'response_time_p95',
        threshold: 1000,
        severity: 'warning'
      },
      {
        metric: 'error_rate',
        threshold: 5,
        severity: 'critical'
      },
      {
        metric: 'memory_usage',
        threshold: 90,
        severity: 'warning'
      }
    ];
    
    return alertRules;
  }
}
```

### 2. Automated Performance Testing
```typescript
class AutomatedPerformanceTesting {
  async runPerformanceTests(): Promise<TestResults> {
    const tests = [
      {
        name: 'API Load Test',
        test: () => this.runAPILoadTest(),
        threshold: { responseTime: 500, throughput: 1000 }
      },
      {
        name: 'Database Performance Test',
        test: () => this.runDatabaseTest(),
        threshold: { queryTime: 100, throughput: 500 }
      },
      {
        name: 'Memory Usage Test',
        test: () => this.runMemoryTest(),
        threshold: { memoryUsage: 80, leakRate: 0 }
      }
    ];
    
    const results = [];
    for (const test of tests) {
      const result = await test.test();
      const passed = this.evaluateThreshold(result, test.threshold);
      results.push({ ...test, result, passed });
    }
    
    return results;
  }
  
  private evaluateThreshold(result: any, threshold: any): boolean {
    return Object.keys(threshold).every(key => 
      result[key] <= threshold[key]
    );
  }
}
```

This comprehensive examples file provides practical, real-world scenarios that demonstrate how to use the MCP Optimizer mode effectively across different technologies and optimization challenges.
