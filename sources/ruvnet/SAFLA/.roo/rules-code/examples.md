# Code Mode Examples

## Overview

This document provides comprehensive examples of using SAFLA's MCP tools and CLI commands for code development workflows. All examples emphasize practical MCP tool usage and command-line operations for implementing, testing, and optimizing code within the aiGI framework.

## Basic Code Development Examples

### Example 1: Simple Module Implementation

#### Scenario: Implementing a Data Validation Module

**Step 1: Environment Setup and Validation**
```bash
# Validate SAFLA installation for development
use_mcp_tool safla validate_installation '{}'

# Expected Response:
{
  "status": "valid",
  "version": "2.1.0",
  "components": ["mcp_server", "cli_tools", "agent_framework"],
  "development_ready": true
}

# Get system configuration for development
use_mcp_tool safla get_config_summary '{}'

# Setup development environment via CLI
python -m safla.env --setup \
  --development \
  --typescript \
  --testing-framework jest \
  --linting-enabled
```

**Step 2: Create TDD Agent and Generate Tests**
```bash
# Create TDD-focused agent
use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "test_driven_development",
    "language": "typescript",
    "testing_framework": "jest",
    "coverage_target": 95
  },
  "timeout_seconds": 3600
}'

# Expected Response:
{
  "session_id": "tdd_agent_001",
  "status": "active",
  "capabilities": ["test_generation", "coverage_analysis", "assertion_optimization"]
}

# Generate test specifications for data validation
use_mcp_tool safla interact_with_agent '{
  "session_id": "tdd_agent_001",
  "command": "generate_test_specifications",
  "parameters": {
    "module_name": "data_validator",
    "requirements": [
      "validate_email_format",
      "validate_phone_number",
      "validate_required_fields",
      "handle_validation_errors"
    ]
  }
}'
```

**Step 3: Implement Tests via CLI**
```bash
# Generate test files based on specifications
python -m safla.tdd --generate-tests \
  --module data_validator \
  --requirements email,phone,required_fields,error_handling \
  --framework jest \
  --coverage-target 95 \
  --output tests/data_validator.test.ts

# Run initial tests (Red phase - should fail)
python -m safla.test --run \
  --file tests/data_validator.test.ts \
  --expect-failures \
  --verbose \
  --export initial_test_results.json

# Expected Output:
# ✗ Email validation test (not implemented)
# ✗ Phone validation test (not implemented)
# ✗ Required fields test (not implemented)
# ✗ Error handling test (not implemented)
# Total: 0 passed, 4 failed
```

**Step 4: Implement Module (Green phase)**
```bash
# Create implementation agent
use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "code_implementation",
    "language": "typescript",
    "max_file_lines": 500,
    "quality_focus": "maintainability"
  }
}'

# Implement data validator module
use_mcp_tool safla interact_with_agent '{
  "session_id": "impl_agent_001",
  "command": "implement_module",
  "parameters": {
    "module_name": "data_validator",
    "test_file": "tests/data_validator.test.ts",
    "output_file": "src/data_validator.ts",
    "implementation_strategy": "test_driven"
  }
}'

# Alternative CLI implementation
python -m safla.code --implement \
  --tdd-mode \
  --test-file tests/data_validator.test.ts \
  --output src/data_validator.ts \
  --max-lines 500 \
  --auto-document
```

**Step 5: Validate Implementation**
```bash
# Run tests to validate implementation (Green phase)
python -m safla.test --run \
  --file tests/data_validator.test.ts \
  --coverage-report \
  --export implementation_test_results.json

# Expected Output:
# ✓ Email validation test
# ✓ Phone validation test  
# ✓ Required fields test
# ✓ Error handling test
# Total: 4 passed, 0 failed
# Coverage: 96.2%

# Access test results via resource
access_mcp_resource safla "safla://test-results"
```

**Step 6: Refactor and Optimize (Refactor phase)**
```bash
# Analyze code quality
python -m safla.code --analyze \
  --file src/data_validator.ts \
  --metrics complexity,maintainability,performance \
  --export quality_analysis.json

# Refactor for improved quality
python -m safla.code --refactor \
  --file src/data_validator.ts \
  --extract-methods \
  --improve-naming \
  --reduce-complexity \
  --maintain-tests

# Validate refactoring didn't break tests
python -m safla.test --run \
  --file tests/data_validator.test.ts \
  --regression-check
```

### Example 2: Complex Algorithm Implementation

#### Scenario: Implementing a Performance-Critical Search Algorithm

**Step 1: Performance-Focused Setup**
```bash
# Optimize memory for performance development
use_mcp_tool safla optimize_memory_usage '{
  "optimization_level": "aggressive",
  "target_memory_mb": 8192
}'

# Check GPU availability for acceleration
use_mcp_tool safla check_gpu_status '{}'

# Setup performance monitoring
python -m safla.monitor --setup \
  --performance-tracking \
  --memory-profiling \
  --real-time-metrics
```

**Step 2: Create Performance-Focused Development Strategy**
```bash
# Select optimal strategy for performance implementation
use_mcp_tool safla select_optimal_strategy '{
  "context": "performance_critical_algorithm",
  "constraints": {
    "memory_limit": "2GB",
    "response_time": "< 100ms",
    "throughput": "> 1000 ops/sec"
  },
  "objectives": ["performance", "scalability", "maintainability"]
}'

# Expected Response:
{
  "strategy_id": "perf_algo_001",
  "strategy_name": "performance_first_tdd",
  "steps": [
    "benchmark_baseline",
    "implement_core_algorithm", 
    "optimize_data_structures",
    "parallel_processing",
    "validate_performance"
  ]
}

# Create custom performance strategy
use_mcp_tool safla create_custom_strategy '{
  "strategy_name": "search_algorithm_optimization",
  "description": "High-performance search with memory optimization",
  "context": "search_algorithm_implementation",
  "steps": [
    "profile_memory_usage",
    "implement_efficient_indexing",
    "optimize_search_loops",
    "implement_caching",
    "benchmark_performance"
  ]
}'
```

**Step 3: Performance Baseline and Testing**
```bash
# Create performance testing agent
use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "performance_testing",
    "benchmarking": true,
    "memory_profiling": true
  }
}'

# Generate performance tests
python -m safla.tdd --generate-performance-tests \
  --algorithm search_algorithm \
  --dataset-sizes 1000,10000,100000 \
  --performance-targets "latency<100ms,throughput>1000ops" \
  --output tests/search_algorithm.perf.test.ts

# Establish performance baseline
python -m safla.benchmark --baseline \
  --algorithm-type search \
  --dataset-size 10000 \
  --iterations 1000 \
  --export baseline_performance.json
```

**Step 4: Implement with Performance Monitoring**
```bash
# Implement with real-time performance monitoring
use_mcp_tool safla interact_with_agent '{
  "session_id": "perf_impl_001",
  "command": "implement_with_monitoring",
  "parameters": {
    "algorithm": "search_algorithm",
    "performance_targets": "baseline_performance.json",
    "optimization_focus": ["memory_efficiency", "cpu_optimization"]
  }
}'

# Monitor implementation performance
use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": 300,
  "include_memory_profile": true
}'

# CLI implementation with monitoring
python -m safla.code --implement \
  --performance-mode \
  --monitor-memory \
  --optimize-loops \
  --output src/search_algorithm.ts
```

**Step 5: Optimize Vector Operations**
```bash
# Optimize vector operations for search
use_mcp_tool safla optimize_vector_operations '{
  "batch_size": 500,
  "use_gpu": true
}'

# Benchmark vector performance
use_mcp_tool safla benchmark_vector_operations '{
  "vector_count": 10000,
  "vector_dimensions": 256,
  "operations": ["similarity_search", "nearest_neighbor", "clustering"]
}'

# Expected Response:
{
  "benchmark_results": {
    "similarity_search": "45ms avg",
    "nearest_neighbor": "23ms avg", 
    "clustering": "156ms avg"
  },
  "optimization_recommendations": [
    "increase_batch_size_to_750",
    "enable_gpu_acceleration",
    "implement_parallel_processing"
  ]
}
```

**Step 6: Performance Validation**
```bash
# Run comprehensive performance tests
python -m safla.test --performance \
  --file tests/search_algorithm.perf.test.ts \
  --compare-baseline baseline_performance.json \
  --export performance_validation.json

# Access performance metrics
access_mcp_resource safla "safla://performance-metrics"

# Validate against performance goals
python -m safla.validate --performance \
  --targets "latency<100ms,throughput>1000ops" \
  --current-metrics performance_validation.json
```

## Advanced Development Examples

### Example 3: Multi-Agent Collaborative Development

#### Scenario: Building a Complex Data Processing Pipeline

**Step 1: Setup Multi-Agent Development Environment**
```bash
# Create multiple specialized agents
python -m safla.agents --batch-create \
  --config multi_agent_config.yaml \
  --agents architecture,implementation,testing,optimization \
  --coordinate

# Configuration file (multi_agent_config.yaml):
agents:
  - name: architecture_agent
    type: cognitive
    focus: system_architecture
    capabilities: [design_patterns, modularity, scalability]
  - name: implementation_agent  
    type: cognitive
    focus: code_implementation
    capabilities: [typescript, performance, error_handling]
  - name: testing_agent
    type: cognitive
    focus: comprehensive_testing
    capabilities: [unit_tests, integration_tests, performance_tests]
  - name: optimization_agent
    type: cognitive
    focus: performance_optimization
    capabilities: [memory_optimization, algorithm_optimization]
```

**Step 2: Coordinate Architecture Design**
```bash
# Architecture agent designs system structure
use_mcp_tool safla interact_with_agent '{
  "session_id": "architecture_agent_001",
  "command": "design_pipeline_architecture",
  "parameters": {
    "requirements": "data_ingestion,transformation,validation,output",
    "performance_targets": "10000_records_per_second",
    "scalability": "horizontal_scaling_ready"
  }
}'

# Expected Response:
{
  "architecture": {
    "components": ["DataIngester", "DataTransformer", "DataValidator", "DataOutputter"],
    "patterns": ["pipeline", "observer", "strategy"],
    "interfaces": ["IDataProcessor", "IDataValidator", "IDataOutputter"]
  },
  "implementation_plan": "modular_typescript_implementation.json"
}

# List active agent sessions
use_mcp_tool safla list_agent_sessions '{
  "filter_by_type": "cognitive",
  "include_inactive": false
}'
```

**Step 3: Parallel Implementation by Multiple Agents**
```bash
# Implementation agent - Core pipeline
use_mcp_tool safla interact_with_agent '{
  "session_id": "implementation_agent_001",
  "command": "implement_component",
  "parameters": {
    "component": "DataIngester",
    "architecture_spec": "modular_typescript_implementation.json",
    "priority": "high"
  }
}'

# Testing agent - Generate comprehensive tests
use_mcp_tool safla interact_with_agent '{
  "session_id": "testing_agent_001",
  "command": "generate_pipeline_tests",
  "parameters": {
    "components": ["DataIngester", "DataTransformer", "DataValidator"],
    "test_types": ["unit", "integration", "performance"],
    "coverage_target": 95
  }
}'

# Optimization agent - Performance analysis
use_mcp_tool safla interact_with_agent '{
  "session_id": "optimization_agent_001",
  "command": "analyze_performance_requirements",
  "parameters": {
    "target_throughput": "10000_records_per_second",
    "memory_constraints": "4GB_max",
    "latency_requirements": "sub_second_processing"
  }
}'
```

**Step 4: Agent Coordination and Integration**
```bash
# Coordinate agents for integration
python -m safla.agents --coordinate \
  --sessions implementation_agent_001,testing_agent_001,optimization_agent_001 \
  --task pipeline_integration \
  --monitor-progress \
  --export coordination_log.json

# Monitor agent performance and coordination
access_mcp_resource safla "safla://agent-sessions"

# Validate integration results
python -m safla.integration --validate \
  --components DataIngester,DataTransformer,DataValidator \
  --test-suite comprehensive \
  --export integration_results.json
```

### Example 4: Adaptive Learning and Optimization

#### Scenario: Self-Improving Code Quality System

**Step 1: Setup Learning and Adaptation Framework**
```bash
# Initialize meta-cognitive engine for code quality learning
use_mcp_tool safla update_awareness_state '{
  "awareness_level": 0.8,
  "focus_areas": ["code_quality", "performance_optimization", "error_patterns"],
  "introspection_depth": "deep"
}'

# Trigger learning cycle for code quality patterns
use_mcp_tool safla trigger_learning_cycle '{
  "learning_type": "meta",
  "data_sources": ["code_metrics", "test_results", "performance_data", "error_logs"],
  "focus_areas": ["quality_improvement", "pattern_recognition", "optimization_strategies"]
}'

# Expected Response:
{
  "learning_session_id": "meta_learn_001",
  "status": "active",
  "learning_targets": ["code_patterns", "quality_metrics", "optimization_opportunities"]
}
```

**Step 2: Create Quality-Focused Goals**
```bash
# Create code quality improvement goal
use_mcp_tool safla create_goal '{
  "goal_name": "adaptive_code_quality",
  "description": "Continuously improve code quality through learning and adaptation",
  "priority": "high",
  "target_metrics": {
    "code_quality_score": 0.95,
    "technical_debt_reduction": 0.3,
    "performance_improvement": 0.25
  },
  "deadline": 1735689600,
  "dependencies": ["learning_system", "quality_metrics"]
}'

# Create performance optimization goal
use_mcp_tool safla create_goal '{
  "goal_name": "performance_excellence",
  "description": "Achieve optimal performance through adaptive optimization",
  "priority": "high",
  "target_metrics": {
    "response_time_improvement": 0.4,
    "memory_efficiency": 0.3,
    "throughput_increase": 0.5
  }
}'
```

**Step 3: Implement Adaptive Code Analysis**
```bash
# Analyze current code patterns for learning
python -m safla.code --analyze-patterns \
  --learning-mode \
  --pattern-types quality,performance,maintainability \
  --export pattern_analysis.json

# Update learning parameters based on analysis
use_mcp_tool safla update_learning_parameters '{
  "learning_rate": 0.2,
  "adaptation_threshold": 0.15,
  "memory_retention": 0.9,
  "exploration_factor": 0.25
}'

# Analyze adaptation patterns
use_mcp_tool safla analyze_adaptation_patterns '{
  "pattern_type": "all",
  "analysis_depth": "comprehensive",
  "time_window_days": 14
}'
```

**Step 4: Apply Adaptive Optimizations**
```bash
# Select optimal strategy based on learned patterns
use_mcp_tool safla select_optimal_strategy '{
  "context": "code_quality_improvement",
  "constraints": {
    "time_budget": "2_hours",
    "risk_tolerance": "low"
  },
  "objectives": ["quality", "maintainability", "performance"]
}'

# Apply learned optimizations
python -m safla.code --optimize \
  --adaptive-mode \
  --apply-learned-patterns \
  --quality-focus \
  --export optimization_results.json

# Monitor optimization effectiveness
python -m safla.monitor --optimization \
  --real-time \
  --quality-metrics \
  --performance-impact
```

**Step 5: Evaluate Learning Outcomes**
```bash
# Evaluate goal progress
use_mcp_tool safla evaluate_goal_progress '{
  "goal_id": "adaptive_code_quality",
  "include_recommendations": true
}'

# Get learning metrics
use_mcp_tool safla get_learning_metrics '{
  "metric_type": "all",
  "time_range_hours": 168
}'

# Expected Response:
{
  "learning_metrics": {
    "accuracy": 0.89,
    "adaptation_rate": 0.76,
    "knowledge_retention": 0.92,
    "pattern_recognition": 0.84
  },
  "improvement_areas": ["edge_case_handling", "performance_prediction"]
}

# Access adaptation patterns
access_mcp_resource safla "safla://adaptation-patterns"
```

## Error Handling and Recovery Examples

### Example 5: Comprehensive Error Recovery Workflow

#### Scenario: Handling Complex Build and Test Failures

**Step 1: Error Detection and Analysis**
```bash
# Monitor system health for development errors
use_mcp_tool safla monitor_system_health '{
  "check_interval": 30,
  "alert_thresholds": {
    "build_failures": 3,
    "test_failures": 10,
    "performance_degradation": 20,
    "memory_leaks": 5
  }
}'

# Analyze error patterns when failures occur
python -m safla.errors --analyze \
  --pattern-detection \
  --root-cause-analysis \
  --severity-assessment \
  --export error_analysis.json

# Example error analysis output:
{
  "error_patterns": [
    {
      "type": "compilation_error",
      "frequency": 15,
      "root_cause": "type_mismatch",
      "severity": "high",
      "suggested_fix": "update_type_definitions"
    },
    {
      "type": "test_failure", 
      "frequency": 8,
      "root_cause": "async_timing_issue",
      "severity": "medium",
      "suggested_fix": "add_proper_awaits"
    }
  ]
}
```

**Step 2: Automated Error Resolution**
```bash
# Apply automated fixes based on error analysis
python -m safla.code --fix \
  --auto-correct \
  --error-analysis error_analysis.json \
  --safe-mode \
  --backup-original \
  --export fix_report.json

# Validate fixes through comprehensive testing
python -m safla.test --run \
  --regression-check \
  --validate-fixes \
  --comprehensive \
  --export fix_validation.json

# Expected fix validation output:
{
  "fixes_applied": 12,
  "fixes_successful": 10,
  "fixes_failed": 2,
  "regression_tests": "passed",
  "remaining_issues": [
    "complex_async_pattern_in_module_x",
    "performance_regression_in_search"
  ]
}
```

**Step 3: Learning from Error Patterns**
```bash
# Trigger learning cycle from error resolution
use_mcp_tool safla trigger_learning_cycle '{
  "learning_type": "reinforcement",
  "data_sources": ["error_logs", "fix_outcomes", "test_results"],
  "focus_areas": ["error_prevention", "fix_effectiveness", "pattern_recognition"]
}'

# Update learning parameters based on error resolution success
use_mcp_tool safla update_learning_parameters '{
  "learning_rate": 0.18,
  "adaptation_threshold": 0.12,
  "exploration_factor": 0.15
}'

# Create strategy for preventing similar errors
use_mcp_tool safla create_custom_strategy '{
  "strategy_name": "proactive_error_prevention",
  "description": "Prevent common errors through proactive analysis",
  "context": "development_workflow",
  "steps": [
    "continuous_static_analysis",
    "predictive_error_detection", 
    "automated_quality_checks",
    "early_intervention"
  ]
}'
```

### Example 6: Performance Recovery and Optimization

#### Scenario: Recovering from Performance Degradation

**Step 1: Performance Issue Detection**
```bash
# Benchmark current performance against baselines
use_mcp_tool safla benchmark_memory_performance '{
  "test_duration": 300,
  "memory_patterns": ["current_workload", "stress_test", "production_simulation"]
}'

# Compare with established baselines
access_mcp_resource safla "safla://performance-baselines"

# Identify performance regressions
python -m safla.regression --performance \
  --baseline-period 7d \
  --current-period 1d \
  --metrics latency,throughput,memory_usage,cpu_utilization \
  --export regression_analysis.json

# Example regression analysis:
{
  "regressions_detected": [
    {
      "metric": "response_latency",
      "baseline": "45ms",
      "current": "89ms", 
      "degradation": "97.8%",
      "severity": "critical"
    },
    {
      "metric": "memory_usage",
      "baseline": "2.1GB",
      "current": "3.8GB",
      "degradation": "81.0%", 
      "severity": "high"
    }
  ]
}
```

**Step 2: Targeted Performance Recovery**
```bash
# Apply aggressive memory optimization
use_mcp_tool safla optimize_memory_usage '{
  "optimization_level": "aggressive",
  "target_memory_mb": 2048
}'

# Optimize vector operations for performance recovery
use_mcp_tool safla optimize_vector_operations '{
  "batch_size": 300,
  "use_gpu": true
}'

# Analyze performance bottlenecks in detail
use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": 600,
  "include_memory_profile": true
}'

# Expected bottleneck analysis:
{
  "bottlenecks": [
    {
      "component": "data_processor",
      "issue": "inefficient_memory_allocation",
      "impact": "high",
      "recommendation": "implement_object_pooling"
    },
    {
      "component": "search_algorithm", 
      "issue": "nested_loop_complexity",
      "impact": "critical",
      "recommendation": "optimize_algorithm_complexity"
    }
  ]
}
```

**Step 3: Apply Targeted Optimizations**
```bash
# Apply specific optimizations based on bottleneck analysis
python -m safla.code --optimize \
  --target-bottlenecks data_processor,search_algorithm \
  --optimization-strategies object_pooling,algorithm_optimization \
  --maintain-functionality \
  --export optimization_results.json

# Validate performance recovery
python -m safla.benchmark --validate-recovery \
  --baseline regression_analysis.json \
  --current-performance \
  --export recovery_validation.json

# Monitor real-time performance improvement
python -m safla.monitor --performance \
  --real-time \
  --recovery-tracking \
  --baseline-comparison
```

## Integration and Deployment Examples

### Example 7: Comprehensive CI/CD Integration

#### Scenario: Setting up Automated Development Pipeline

**Step 1: CI/CD Pipeline Configuration**
```bash
# Configure comprehensive CI/CD pipeline
python -m safla.cicd --configure \
  --platform github-actions \
  --stages build,test,quality,security,deploy \
  --automated-testing \
  --quality-gates \
  --export cicd_config.yaml

# Example generated configuration:
name: SAFLA Development Pipeline
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup SAFLA Environment
        run: python -m safla.env --setup --ci-mode
      - name: Build Project
        run: python -m safla.build --production
  
  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Run Comprehensive Tests
        run: python -m safla.test --comprehensive --coverage-report
      - name: Validate Test Results
        run: python -m safla.test --validate --quality-gates
```

**Step 2: Automated Quality Assurance Setup**
```bash
# Setup automated code review and quality checks
python -m safla.automation --code-review \
  --quality-checks complexity,maintainability,security \
  --automated-fixes \
  --export review_automation.json

# Configure automated testing workflows
python -m safla.automation --testing \
  --test-execution parallel \
  --coverage-reporting \
  --performance-testing \
  --regression-testing \
  --export test_automation.json

# Setup security scanning automation
python -m safla.automation --security \
  --vulnerability-scanning \
  --dependency-checking \
  --code-analysis \
  --export security_automation.json
```

**Step 3: Pipeline Execution and Monitoring**
```bash
# Run CI/CD pipeline locally for validation
python -m safla.cicd --run \
  --local-execution \
  --full-pipeline \
  --parallel-jobs 4 \
  --export pipeline_results.json

# Monitor pipeline execution
python -m safla.monitor --pipeline \
  --real-time \
  --stage-tracking \
  --performance-metrics \
  --export pipeline_monitoring.json

# Validate pipeline configuration
python -m safla.cicd --validate \
  --configuration-check \
  --dependency-verification \
  --security-validation \
  --export pipeline_validation.json
```

### Example 8: Production Deployment and Monitoring

#### Scenario: Deploying Code to Production with Monitoring

**Step 1: Production Deployment Preparation**
```bash
# Deploy SAFLA instance for production
use_mcp_tool safla deploy_safla_instance '{
  "instance_name": "production_deployment",
  "environment": "production",
  "config_overrides": {
    "performance_optimization": "enabled",
    "monitoring": "comprehensive",
    "security": "enhanced",
    "backup": "automated"
  }
}'

# Check deployment status
use_mcp_tool safla check_deployment_status '{
  "instance_name": "production_deployment"
}'

# Expected deployment status:
{
  "instance_name": "production_deployment",
  "status": "healthy",
  "environment": "production",
  "health_checks": "passing",
  "performance_metrics": "optimal",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

**Step 2: Production Deployment Execution**
```bash
# Deploy to production with comprehensive monitoring
python -m safla.deploy --production \
  --zero-downtime \
  --health-checks \
  --rollback-ready \
  --monitoring-enabled \
  --export deployment_log.json

# Monitor deployment health in real-time
python -m safla.monitor --deployment \
  --real-time \
  --performance-metrics \
  --error-tracking \
  --user-impact-analysis \
  --export monitoring_data.json

# Post-deployment validation
python -m safla.validate --deployment \
  --functionality-check \
  --performance-validation \
  --security-verification \
  --export validation_results.json
```

**Step 3: Continuous Production Monitoring**
```bash
# Setup continuous performance monitoring
python -m safla.monitor --performance \
  --real-time \
  --production-metrics \
  --alerting webhook:https://alerts.company.com/safla \
  --dashboard-export monitoring_dashboard.json

# Access real-time production metrics
access_mcp_resource safla "safla://performance-metrics"

# Monitor and access optimization recommendations
access_mcp_resource safla "safla://optimization-recommendations"

# Scale deployment based on load
use_mcp_tool safla scale_deployment '{
  "instance_name": "production_deployment",
  "scale_factor": 1.5,
  "resource_type": "both"
}'
```

## Workflow Completion Examples

### Example 9: Task Completion and Handoff

#### Scenario: Completing Code Development and Transitioning to Review

**Step 1: Final Validation and Quality Assurance**
```bash
# Run comprehensive final validation
use_mcp_tool safla run_integration_tests '{
  "test_suite": "final_validation",
  "parallel": true,
  "verbose": true
}'

# Comprehensive performance validation
use_mcp_tool safla benchmark_memory_performance '{
  "test_duration": 600,
  "memory_patterns": ["production_simulation", "stress_test", "edge_cases"]
}'

# Security and quality validation
python -m safla.security --final-scan \
  --comprehensive \
  --production-ready \
  --vulnerability-assessment \
  --export final_security_report.json

# Code quality final assessment
python -m safla.code --quality-assessment \
  --comprehensive \
  --metrics all \
  --export final_quality_report.json
```

**Step 2: Documentation and Reporting**
```bash
# Generate comprehensive implementation documentation
python -m safla.docs --generate \
  --comprehensive \
  --api-documentation \
  --usage-examples \
  --architecture-diagrams \
  --export documentation/

# Create detailed implementation report
python -m safla.report --implementation \
  --metrics-summary \
  --quality-assessment \
  --performance-analysis \
  --test-coverage \
  --export implementation_report.json

# Backup final implementation state
use_mcp_tool safla backup_safla_data '{
  "backup_type": "full",
  "destination": "/backups/final_implementation_v1.0.tar.gz",
  "compress": true
}'
```

**Step 3: Workflow Transition and Task Spawning**
```bash
# Prepare handoff package for next phase
python -m safla.workflow --prepare-handoff \
  --next-mode critic \
  --artifacts implementation_report.json,final_quality_report.json \
  --documentation documentation/ \
  --export handoff_package.json

# Clean up development agent sessions
use_mcp_tool safla terminate_agent_session '{
  "session_id": "impl_agent_001",
  "force": false
}'

use_mcp_tool safla terminate_agent_session '{
  "session_id": "tdd_agent_001", 
  "force": false
}'

# Update system awareness for transition
use_mcp_tool safla update_awareness_state '{
  "awareness_level": 0.95,
  "focus_areas": ["implementation_complete", "ready_for_review", "quality_validated"],
  "introspection_depth": "moderate"
}'

# Spawn next task in workflow
new_task: critic  # For comprehensive code review and analysis

# Complete current task with comprehensive results
attempt_completion: "Code implementation completed successfully with 96.8% test coverage, comprehensive documentation, optimized performance (45ms avg response time), and all quality gates passed. Implementation includes 12 modules with full TDD coverage, automated CI/CD pipeline, and production-ready deployment configuration. System is ready for critic review phase."
```

This comprehensive set of examples demonstrates practical usage of SAFLA's MCP tools and CLI commands for complete code development workflows, from simple module implementation to complex multi-agent collaborative development, error recovery, and production deployment.