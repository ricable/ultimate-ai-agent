# TDD Mode Examples Documentation

## Overview

This document provides comprehensive practical examples for implementing Test-Driven Development workflows using SAFLA's MCP tools and CLI commands. Each example demonstrates real-world scenarios with complete command sequences, expected outputs, and best practices for effective TDD implementation.

## Example 1: User Authentication System TDD

### Scenario: Implementing a secure user authentication system using comprehensive TDD approach

#### Phase 1: Setup and Planning
```bash
# Initialize TDD session with system awareness
$ use_mcp_tool safla get_system_awareness '{}'
{
  "awareness_state": {
    "current_focus": "test_driven_development",
    "system_health": "optimal",
    "resource_availability": "high",
    "learning_readiness": true
  }
}

# Create comprehensive testing goal
$ use_mcp_tool safla create_goal '{
  "goal_name": "auth_system_tdd_excellence",
  "description": "Implement secure authentication system with 95% test coverage",
  "priority": "high",
  "target_metrics": {
    "line_coverage": 0.95,
    "branch_coverage": 0.92,
    "security_score": 0.98,
    "performance_score": 0.90
  },
  "deadline": 1706745600
}'
{
  "goal_id": "auth_tdd_001",
  "status": "created",
  "estimated_completion": "2024-01-31T16:00:00Z"
}

# Select optimal TDD strategy
$ use_mcp_tool safla select_optimal_strategy '{
  "context": "secure_authentication_development",
  "constraints": {
    "security_requirements": "high",
    "performance_requirements": "medium",
    "time_constraint": "2_weeks"
  },
  "objectives": ["comprehensive_coverage", "security_validation", "performance_optimization"]
}'
{
  "selected_strategy": {
    "strategy_id": "security_focused_tdd_001",
    "strategy_name": "Security-Focused TDD with Performance Validation",
    "effectiveness_score": 94.2,
    "components": [
      "security_test_first",
      "comprehensive_edge_case_coverage",
      "performance_baseline_establishment",
      "continuous_security_validation"
    ]
  }
}
```

#### Phase 2: Agent Setup and Test Planning
```bash
# Create cognitive agent for TDD assistance
$ use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "security_tdd",
    "optimization_level": "comprehensive",
    "learning_enabled": true,
    "security_awareness": true
  },
  "timeout_seconds": 14400
}'
{
  "session_id": "security_tdd_agent_001",
  "status": "active",
  "capabilities": ["test_generation", "security_analysis", "performance_validation"]
}

# Generate comprehensive test specifications
$ use_mcp_tool safla interact_with_agent '{
  "session_id": "security_tdd_agent_001",
  "command": "generate_security_test_plan",
  "parameters": {
    "component": "user_authentication",
    "security_requirements": [
      "password_hashing",
      "session_management", 
      "rate_limiting",
      "input_validation",
      "sql_injection_prevention"
    ],
    "test_categories": ["unit", "integration", "security", "performance"]
  }
}'
{
  "test_plan": {
    "total_tests": 47,
    "security_tests": 18,
    "performance_tests": 8,
    "integration_tests": 12,
    "unit_tests": 9,
    "estimated_coverage": "96.2%"
  },
  "test_specifications": [
    {
      "test_name": "password_hashing_security",
      "category": "security",
      "description": "Validate bcrypt hashing with proper salt rounds",
      "assertions": ["hash_uniqueness", "salt_randomness", "timing_resistance"]
    },
    {
      "test_name": "session_token_validation",
      "category": "security", 
      "description": "Validate JWT token generation and validation",
      "assertions": ["token_expiry", "signature_validation", "payload_integrity"]
    }
  ]
}
```

#### Phase 3: Red Phase - Writing Failing Tests
```bash
# Create test files based on specifications
$ cat > tests/auth.security.test.ts << 'EOF'
import { AuthService } from '../src/auth/AuthService';
import { SecurityValidator } from '../src/auth/SecurityValidator';

describe('Authentication Security Tests', () => {
  let authService: AuthService;
  let securityValidator: SecurityValidator;

  beforeEach(() => {
    authService = new AuthService();
    securityValidator = new SecurityValidator();
  });

  describe('Password Hashing Security', () => {
    test('should hash passwords with bcrypt and proper salt rounds', async () => {
      const password = 'testPassword123!';
      const hash = await authService.hashPassword(password);
      
      expect(hash).toBeDefined();
      expect(hash).not.toBe(password);
      expect(hash.startsWith('$2b$')).toBe(true);
      expect(await authService.verifyPassword(password, hash)).toBe(true);
    });

    test('should generate unique hashes for identical passwords', async () => {
      const password = 'samePassword123!';
      const hash1 = await authService.hashPassword(password);
      const hash2 = await authService.hashPassword(password);
      
      expect(hash1).not.toBe(hash2);
    });

    test('should resist timing attacks', async () => {
      const validPassword = 'validPassword123!';
      const invalidPassword = 'invalidPassword123!';
      const hash = await authService.hashPassword(validPassword);
      
      const startValid = Date.now();
      await authService.verifyPassword(validPassword, hash);
      const validTime = Date.now() - startValid;
      
      const startInvalid = Date.now();
      await authService.verifyPassword(invalidPassword, hash);
      const invalidTime = Date.now() - startInvalid;
      
      // Timing difference should be minimal (< 10ms)
      expect(Math.abs(validTime - invalidTime)).toBeLessThan(10);
    });
  });

  describe('Session Management Security', () => {
    test('should generate secure JWT tokens with proper expiry', async () => {
      const user = { id: 1, email: 'test@example.com' };
      const token = await authService.generateToken(user);
      
      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      
      const decoded = await authService.verifyToken(token);
      expect(decoded.id).toBe(user.id);
      expect(decoded.exp).toBeGreaterThan(Date.now() / 1000);
    });

    test('should reject expired tokens', async () => {
      const user = { id: 1, email: 'test@example.com' };
      const expiredToken = await authService.generateToken(user, -3600); // Expired 1 hour ago
      
      await expect(authService.verifyToken(expiredToken)).rejects.toThrow('Token expired');
    });
  });
});
EOF

# Run initial tests (expecting failures)
$ npm test -- --testPathPattern="auth.security" --verbose
FAIL tests/auth.security.test.ts
  Authentication Security Tests
    Password Hashing Security
      ✕ should hash passwords with bcrypt and proper salt rounds (2 ms)
      ✕ should generate unique hashes for identical passwords (1 ms)
      ✕ should resist timing attacks (1 ms)
    Session Management Security
      ✕ should generate secure JWT tokens with proper expiry (1 ms)
      ✕ should reject expired tokens (1 ms)

Test Suites: 1 failed, 1 total
Tests:       5 failed, 5 total

# Monitor test results through SAFLA
$ access_mcp_resource safla "safla://test-results"
{
  "latest_execution": {
    "timestamp": "2024-01-15T14:30:00Z",
    "total_tests": 5,
    "passed": 0,
    "failed": 5,
    "execution_time": "0.8s",
    "failure_categories": {
      "module_not_found": 5,
      "implementation_missing": 5
    }
  }
}

# Analyze failures with agent assistance
$ use_mcp_tool safla interact_with_agent '{
  "session_id": "security_tdd_agent_001",
  "command": "analyze_test_failures",
  "parameters": {
    "failure_type": "implementation_missing",
    "test_category": "security",
    "suggest_implementation_order": true
  }
}'
{
  "analysis": {
    "failure_cause": "Missing AuthService and SecurityValidator implementations",
    "implementation_priority": [
      "AuthService.hashPassword",
      "AuthService.verifyPassword", 
      "AuthService.generateToken",
      "AuthService.verifyToken"
    ],
    "security_considerations": [
      "Use bcrypt with minimum 12 salt rounds",
      "Implement constant-time comparison",
      "Use secure JWT library with proper validation"
    ]
  }
}
```

#### Phase 4: Green Phase - Minimal Implementation
```bash
# Create minimal implementation to pass tests
$ mkdir -p src/auth

$ cat > src/auth/AuthService.ts << 'EOF'
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';

export class AuthService {
  private readonly saltRounds = 12;
  private readonly jwtSecret = process.env.JWT_SECRET || 'default-secret-key';

  async hashPassword(password: string): Promise<string> {
    return bcrypt.hash(password, this.saltRounds);
  }

  async verifyPassword(password: string, hash: string): Promise<boolean> {
    return bcrypt.compare(password, hash);
  }

  async generateToken(user: { id: number; email: string }, expiresInSeconds: number = 3600): Promise<string> {
    const payload = {
      id: user.id,
      email: user.email,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + expiresInSeconds
    };
    
    return jwt.sign(payload, this.jwtSecret);
  }

  async verifyToken(token: string): Promise<any> {
    try {
      return jwt.verify(token, this.jwtSecret);
    } catch (error) {
      if (error.name === 'TokenExpiredError') {
        throw new Error('Token expired');
      }
      throw error;
    }
  }
}
EOF

$ cat > src/auth/SecurityValidator.ts << 'EOF'
export class SecurityValidator {
  // Placeholder for security validation logic
}
EOF

# Install required dependencies
$ npm install bcrypt @types/bcrypt jsonwebtoken @types/jsonwebtoken

# Run tests to verify green phase
$ npm test -- --testPathPattern="auth.security" --verbose
PASS tests/auth.security.test.ts
  Authentication Security Tests
    Password Hashing Security
      ✓ should hash passwords with bcrypt and proper salt rounds (45 ms)
      ✓ should generate unique hashes for identical passwords (89 ms)
      ✓ should resist timing attacks (178 ms)
    Session Management Security
      ✓ should generate secure JWT tokens with proper expiry (12 ms)
      ✓ should reject expired tokens (8 ms)

Test Suites: 1 passed, 1 total
Tests:       5 passed, 5 total

# Monitor successful test execution
$ access_mcp_resource safla "safla://test-results"
{
  "latest_execution": {
    "timestamp": "2024-01-15T14:35:00Z",
    "total_tests": 5,
    "passed": 5,
    "failed": 0,
    "execution_time": "2.1s"
  }
}

# Update goal progress
$ use_mcp_tool safla update_goal '{
  "goal_id": "auth_tdd_001",
  "progress": 0.3,
  "notes": "Green phase completed - basic security tests passing"
}'
```

#### Phase 5: Refactor Phase - Quality Improvement
```bash
# Analyze performance bottlenecks
$ use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": 60,
  "include_memory_profile": true
}'
{
  "bottlenecks": [
    {
      "component": "bcrypt_hashing",
      "impact": "medium",
      "recommendation": "consider_async_optimization"
    }
  ],
  "memory_profile": {
    "peak_usage": "45MB",
    "avg_usage": "32MB",
    "gc_frequency": "normal"
  }
}

# Refactor for better performance and security
$ cat > src/auth/AuthService.ts << 'EOF'
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';

export class AuthService {
  private readonly saltRounds = 12;
  private readonly jwtSecret: string;
  private readonly tokenExpiry = 3600; // 1 hour

  constructor() {
    this.jwtSecret = process.env.JWT_SECRET || this.generateSecureSecret();
  }

  private generateSecureSecret(): string {
    return crypto.randomBytes(64).toString('hex');
  }

  async hashPassword(password: string): Promise<string> {
    this.validatePasswordStrength(password);
    return bcrypt.hash(password, this.saltRounds);
  }

  async verifyPassword(password: string, hash: string): Promise<boolean> {
    // Constant-time comparison to prevent timing attacks
    const isValid = await bcrypt.compare(password, hash);
    
    // Add artificial delay to normalize timing
    const delay = Math.random() * 10;
    await new Promise(resolve => setTimeout(resolve, delay));
    
    return isValid;
  }

  private validatePasswordStrength(password: string): void {
    if (password.length < 8) {
      throw new Error('Password must be at least 8 characters long');
    }
    
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    
    if (!(hasUpperCase && hasLowerCase && hasNumbers && hasSpecialChar)) {
      throw new Error('Password must contain uppercase, lowercase, numbers, and special characters');
    }
  }

  async generateToken(user: { id: number; email: string }, expiresInSeconds: number = this.tokenExpiry): Promise<string> {
    const now = Math.floor(Date.now() / 1000);
    const payload = {
      id: user.id,
      email: user.email,
      iat: now,
      exp: now + expiresInSeconds,
      jti: crypto.randomUUID() // Unique token ID for revocation
    };
    
    return jwt.sign(payload, this.jwtSecret, { algorithm: 'HS256' });
  }

  async verifyToken(token: string): Promise<any> {
    try {
      const decoded = jwt.verify(token, this.jwtSecret, { algorithms: ['HS256'] });
      return decoded;
    } catch (error) {
      if (error.name === 'TokenExpiredError') {
        throw new Error('Token expired');
      }
      if (error.name === 'JsonWebTokenError') {
        throw new Error('Invalid token');
      }
      throw error;
    }
  }
}
EOF

# Run tests to ensure refactoring doesn't break functionality
$ npm test -- --testPathPattern="auth.security" --coverage
PASS tests/auth.security.test.ts
  Authentication Security Tests
    Password Hashing Security
      ✓ should hash passwords with bcrypt and proper salt rounds (52 ms)
      ✓ should generate unique hashes for identical passwords (95 ms)
      ✓ should resist timing attacks (185 ms)
    Session Management Security
      ✓ should generate secure JWT tokens with proper expiry (15 ms)
      ✓ should reject expired tokens (10 ms)

Coverage Summary:
  Lines: 95.2% (40/42)
  Branches: 91.7% (22/24)
  Functions: 100% (8/8)
  Statements: 95.2% (40/42)

# Monitor coverage improvements
$ access_mcp_resource safla "safla://test-coverage"
{
  "coverage_summary": {
    "overall_coverage": 95.2,
    "line_coverage": 95.2,
    "branch_coverage": 91.7,
    "function_coverage": 100.0
  },
  "quality_metrics": {
    "test_quality_score": 92.1,
    "maintainability_index": 87.3,
    "security_score": 94.6
  }
}
```

## Example 2: API Integration Testing with Performance Validation

### Scenario: Testing external API integration with performance benchmarks

#### Setup and Performance Baseline
```bash
# Create performance-focused agent
$ use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "performance_testing",
    "optimization_level": "aggressive",
    "learning_enabled": true
  },
  "timeout_seconds": 3600
}'
{
  "session_id": "perf_test_agent_001",
  "status": "active"
}

# Establish performance baselines
$ use_mcp_tool safla benchmark_memory_performance '{
  "test_duration": 120,
  "memory_patterns": ["api_simulation", "concurrent_requests", "data_processing"]
}'
{
  "baseline_metrics": {
    "avg_response_time": "45ms",
    "throughput": "1250_requests_per_second",
    "memory_efficiency": 94.2,
    "cpu_utilization": 68.4
  }
}

# Create comprehensive API integration tests
$ cat > tests/api.integration.test.ts << 'EOF'
import { ApiClient } from '../src/api/ApiClient';
import { PerformanceMonitor } from '../src/utils/PerformanceMonitor';

describe('API Integration Performance Tests', () => {
  let apiClient: ApiClient;
  let performanceMonitor: PerformanceMonitor;

  beforeEach(() => {
    apiClient = new ApiClient();
    performanceMonitor = new PerformanceMonitor();
  });

  describe('Response Time Performance', () => {
    test('should respond within 100ms for single requests', async () => {
      const startTime = Date.now();
      const response = await apiClient.getUserData(1);
      const responseTime = Date.now() - startTime;
      
      expect(response).toBeDefined();
      expect(responseTime).toBeLessThan(100);
    });

    test('should maintain performance under concurrent load', async () => {
      const concurrentRequests = 50;
      const promises = Array.from({ length: concurrentRequests }, (_, i) => 
        performanceMonitor.measureAsync(() => apiClient.getUserData(i + 1))
      );
      
      const results = await Promise.all(promises);
      const avgResponseTime = results.reduce((sum, result) => sum + result.duration, 0) / results.length;
      
      expect(avgResponseTime).toBeLessThan(150);
      expect(results.every(result => result.success)).toBe(true);
    });
  });

  describe('Memory Efficiency', () => {
    test('should not leak memory during batch operations', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Perform 100 API calls
      for (let i = 0; i < 100; i++) {
        await apiClient.getUserData(i + 1);
      }
      
      // Force garbage collection
      if (global.gc) global.gc();
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024; // MB
      
      expect(memoryIncrease).toBeLessThan(10); // Less than 10MB increase
    });
  });
});
EOF

# Run performance tests
$ npm test -- --testPathPattern="api.integration" --verbose
FAIL tests/api.integration.test.ts
  API Integration Performance Tests
    Response Time Performance
      ✕ should respond within 100ms for single requests
      ✕ should maintain performance under concurrent load
    Memory Efficiency
      ✕ should not leak memory during batch operations

# Implement API client to pass tests
$ mkdir -p src/api src/utils

$ cat > src/api/ApiClient.ts << 'EOF'
import axios, { AxiosInstance } from 'axios';

export class ApiClient {
  private client: AxiosInstance;
  private cache: Map<number, any> = new Map();

  constructor() {
    this.client = axios.create({
      baseURL: process.env.API_BASE_URL || 'https://jsonplaceholder.typicode.com',
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async getUserData(userId: number): Promise<any> {
    // Check cache first for performance
    if (this.cache.has(userId)) {
      return this.cache.get(userId);
    }

    try {
      const response = await this.client.get(`/users/${userId}`);
      
      // Cache the result
      this.cache.set(userId, response.data);
      
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch user data: ${error.message}`);
    }
  }
}
EOF

$ cat > src/utils/PerformanceMonitor.ts << 'EOF'
export class PerformanceMonitor {
  async measureAsync<T>(fn: () => Promise<T>): Promise<{ result: T; duration: number; success: boolean }> {
    const startTime = Date.now();
    
    try {
      const result = await fn();
      const duration = Date.now() - startTime;
      
      return {
        result,
        duration,
        success: true
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      
      return {
        result: null as T,
        duration,
        success: false
      };
    }
  }
}
EOF

# Run tests again
$ npm test -- --testPathPattern="api.integration" --coverage
PASS tests/api.integration.test.ts
  API Integration Performance Tests
    Response Time Performance
      ✓ should respond within 100ms for single requests (67 ms)
      ✓ should maintain performance under concurrent load (234 ms)
    Memory Efficiency
      ✓ should not leak memory during batch operations (156 ms)

# Monitor performance metrics during testing
$ access_mcp_resource safla "safla://performance-metrics"
{
  "test_execution_performance": {
    "avg_test_execution_time": "1.8s",
    "parallel_efficiency": 91.2,
    "resource_utilization": {
      "cpu_usage": 72.1,
      "memory_usage": 1.9,
      "network_io": "moderate"
    }
  },
  "application_performance": {
    "response_times": {
      "avg": "67ms",
      "p95": "89ms",
      "p99": "134ms"
    }
  }
}
```

## Example 3: Continuous Learning and Adaptation

### Scenario: Implementing adaptive test optimization based on learning metrics

```bash
# Initialize learning-focused TDD session
$ use_mcp_tool safla trigger_learning_cycle '{
  "learning_type": "meta",
  "data_sources": ["test_patterns", "failure_analysis", "performance_metrics"],
  "focus_areas": ["test_effectiveness", "coverage_optimization", "performance_prediction"]
}'
{
  "learning_session_id": "meta_learning_001",
  "status": "active",
  "expected_duration": "30_minutes"
}

# Monitor learning progress
$ use_mcp_tool safla get_learning_metrics '{
  "metric_type": "all",
  "time_range_hours": 24
}'
{
  "test_learning_metrics": {
    "test_effectiveness_learning": 87.3,
    "coverage_optimization_learning": 82.1,
    "performance_prediction_accuracy": 94.7,
    "failure_pattern_recognition": 89.2
  },
  "adaptation_success": {
    "successful_optimizations": 23,
    "failed_optimizations": 2,
    "optimization_success_rate": 92.0
  }
}

# Analyze adaptation patterns
$ use_mcp_tool safla analyze_adaptation_patterns '{
  "pattern_type": "all",
  "analysis_depth": "comprehensive",
  "time_window_days": 7
}'
{
  "testing_adaptation_patterns": {
    "test_generation_evolution": {
      "pattern_type": "incremental_improvement",
      "confidence": 94.1,
      "trend": "improving",
      "adaptations": [
        "enhanced_edge_case_detection",
        "improved_assertion_quality",
        "optimized_test_structure"
      ]
    },
    "performance_testing_evolution": {
      "pattern_type": "predictive_optimization",
      "confidence": 91.3,
      "trend": "advancing"
    }
  }
}

# Create adaptive test suite based on learning
$ cat > tests/adaptive.learning.test.ts << 'EOF'
import { LearningEngine } from '../src/learning/LearningEngine';
import { TestOptimizer } from '../src/testing/TestOptimizer';

describe('Adaptive Learning Integration Tests', () => {
  let learningEngine: LearningEngine;
  let testOptimizer: TestOptimizer;

  beforeEach(() => {
    learningEngine = new LearningEngine();
    testOptimizer = new TestOptimizer(learningEngine);
  });

  describe('Test Pattern Learning', () => {
    test('should identify and optimize slow test patterns', async () => {
      const testMetrics = {
        executionTime: 5000,
        memoryUsage: 100,
        complexity: 'high'
      };
      
      const optimization = await testOptimizer.optimizeTest(testMetrics);
      
      expect(optimization.suggestedImprovements).toContain('reduce_complexity');
      expect(optimization.estimatedSpeedup).toBeGreaterThan(0.2);
    });

    test('should adapt test generation based on failure patterns', async () => {
      const failureHistory = [
        { type: 'assertion_error', frequency: 15 },
        { type: 'timeout', frequency: 8 },
        { type: 'memory_leak', frequency: 3 }
      ];
      
      const adaptedStrategy = await learningEngine.adaptTestStrategy(failureHistory);
      
      expect(adaptedStrategy.focusAreas).toContain('assertion_validation');
      expect(adaptedStrategy.timeoutHandling).toBe('enhanced');
    });
  });
});
EOF

# Implement learning components
$ mkdir -p src/learning src/testing

$ cat > src/learning/LearningEngine.ts << 'EOF'
export class LearningEngine {
  private patterns: Map<string, any> = new Map();
  private adaptationHistory: any[] = [];

  async adaptTestStrategy(failureHistory: any[]): Promise<any> {
    const mostCommonFailure = failureHistory.reduce((prev, current) => 
      prev.frequency > current.frequency ? prev : current
    );

    const strategy = {
      focusAreas: [],
      timeoutHandling: 'standard',
      memoryOptimization: 'standard'
    };

    if (mostCommonFailure.type === 'assertion_error') {
      strategy.focusAreas.push('assertion_validation');
    }
    
    if (failureHistory.some(f => f.type === 'timeout')) {
      strategy.timeoutHandling = 'enhanced';
    }

    return strategy;
  }
}
EOF

$ cat > src/testing/TestOptimizer.ts << 'EOF'
import { LearningEngine } from '../learning/LearningEngine';

export class TestOptimizer {
  constructor(private learningEngine: LearningEngine) {}

  async optimizeTest(metrics: any): Promise<any> {
    const optimization = {
      suggestedImprovements: [],
      estimatedSpeedup: 0
    };

    if (metrics.executionTime > 3000) {
      optimization.suggestedImprovements.push('reduce_complexity');
      optimization.estimatedSpeedup += 0.3;
    }

    if (metrics.memoryUsage > 50) {
      optimization.suggestedImprovements.push('optimize_memory');
      optimization.estimatedSpeedup += 0.2;
    }

    return optimization;
  }
}
EOF

# Run adaptive learning tests
$ npm test -- --testPathPattern="adaptive.learning" --verbose
PASS tests/adaptive.learning.test.ts
  Adaptive Learning Integration Tests
    Test Pattern Learning
      ✓ should identify and optimize slow test patterns (23 ms)
      ✓ should adapt test generation based on failure patterns (15 ms)

# Update learning parameters based on success
$ use_mcp_tool safla update_learning_parameters '{
  "learning_rate": 0.15,
  "adaptation_threshold": 0.85,
  "memory_retention": 0.95,
  "exploration_factor": 0.25
}'
{
  "status": "updated",
  "new_parameters": {
    "learning_rate": 0.15,
    "adaptation_threshold": 0.85,
    "memory_retention": 0.95,
    "exploration_factor": 0.25
  }
}
```

## Example 4: Complete TDD Workflow with Goal Completion

### Final Integration and Goal Achievement
```bash
# Run comprehensive test suite
$ use_mcp_tool safla run_integration_tests '{
  "test_suite": "comprehensive",
  "parallel": true,
  "verbose": false
}'
{
  "execution_summary": {
    "total_tests": 47,
    "passed": 47,
    "failed": 0,
    "execution_time": "4m 23s",
    "coverage": {
      "lines": 96.2,
      "branches": 93.1,
      "functions": 100.0
    }
  }
}

# Evaluate final goal progress
$ use_mcp_tool safla evaluate_goal_progress '{
  "goal_id": "auth_tdd_001",
  "include_recommendations": true
}'
{
  "goal_evaluation": {
    "progress": 0.98,
    "target_achievement": {
      "line_coverage": "achieved", // 96.2% vs 95% target
      "branch_coverage": "achieved", // 93.1% vs 92% target
      "security_score": "achieved", // 98.1% vs 98% target
      "performance_score": "exceeded" // 94.2% vs 90% target
    },
    "recommendations": [
      "Consider increasing coverage targets for future goals",
      "Document security testing patterns for reuse"
    ]
  }
}

# Complete goal with final status
$ use_mcp_tool safla update_goal '{
  "goal_id": "auth_tdd_001",
  "status": "completed",
  "progress": 1.0,
  "notes": "TDD workflow completed successfully. Achieved 96.2% line coverage, 93.1% branch coverage, 98.1% security score, and 94.2% performance score."
}'
{
  "status": "completed",
  "completion_time": "2024-01-31T15:45:00Z",
  "final_metrics": {
    "duration": "13_days_15_hours",
    "quality_score": 96.4,
    "efficiency_rating": 92.1
  }
}

# Generate final comprehensive report
$ cat > tdd_completion_report.md << 'EOF'
# TDD Workflow Completion Report

## Goal Achievement Summary
- **Goal**: Secure Authentication System with 95% Test Coverage
- **Status**: ✅ COMPLETED
- **Final Coverage**: 96.2% (exceeded target)
- **Security Score**: 98.1% (met target)
- **Performance Score**: 94.2% (exceeded target)

## Test Execution Results
- **Total Tests**: 47
- **Passed**: 47 (100%)
- **Failed**: 0
- **Execution Time**: 4m 23s

## Learning and Adaptation Metrics
- **Test Effectiveness Learning**: 89.7%
- **Performance Prediction Accuracy**: 94.7%
- **Optimization Success Rate**: 96.2%

## Key Achievements
1. Comprehensive security test coverage
2. Performance optimization through learning
3. Adaptive test strategy implementation
4. Zero test failures in final execution

## Recommendations for Future TDD Cycles
1. Increase coverage targets to 97%
2. Implement automated security scanning
3. Enhance performance baseline tracking
4. Document reusable test patterns
EOF

# Cleanup agent sessions
$ use_mcp_tool safla list_agent_sessions '{}' | jq -r '.active_sessions[].session_id' | while read session_id; do
  use_mcp_tool safla terminate_agent_session '{"session_id": "'$session_id'"}'
done

# Spawn next workflow phase
$ new_task code "Implement production deployment based on TDD-validated authentication system"
```

This comprehensive examples documentation demonstrates practical implementation of TDD workflows using SAFLA's MCP tools and CLI commands, showcasing real-world scenarios with complete command sequences, expected outputs, and best practices for effective test-driven development.