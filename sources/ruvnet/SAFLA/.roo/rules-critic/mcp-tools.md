# Critic Mode MCP Tools Documentation

## Overview

This document provides comprehensive guidance for utilizing SAFLA's MCP tools within the Critic mode for conducting thorough analysis, evaluation, and quality assessment. The Critic mode leverages SAFLA's 40+ MCP tools across system validation, performance analysis, testing, benchmarking, agent coordination, and meta-cognitive capabilities to deliver intelligent, data-driven critiques.

## Core Analysis Tools

### 1. System Validation and Health Assessment Tools

#### System Information and Status Analysis

**Tool**: [`validate_installation`](safla/mcp_stdio_server.py:1)
**Purpose**: Validate system integrity and configuration for comprehensive analysis

```bash
# Validate SAFLA installation for analysis capabilities
use_mcp_tool safla validate_installation '{}'

# Expected Response:
{
  "validation_status": "passed",
  "components": {
    "core_system": "operational",
    "analysis_engine": "active",
    "monitoring_tools": "available",
    "benchmarking_suite": "ready"
  },
  "capabilities": {
    "performance_analysis": true,
    "quality_assessment": true,
    "security_scanning": true,
    "learning_analytics": true
  },
  "recommendations": [
    "All analysis tools operational",
    "System ready for comprehensive critique"
  ]
}
```

**Tool**: [`get_system_info`](safla/mcp_stdio_server.py:1)
**Purpose**: Gather comprehensive system information for analysis context

```bash
# Get detailed system information for analysis baseline
use_mcp_tool safla get_system_info '{}'

# Expected Response:
{
  "system_overview": {
    "version": "SAFLA v2.1.0",
    "environment": "production",
    "uptime": "7d 12h 34m",
    "performance_tier": "high"
  },
  "analysis_capabilities": {
    "static_analysis": "advanced",
    "dynamic_analysis": "comprehensive",
    "security_scanning": "enterprise",
    "performance_profiling": "real_time"
  },
  "resource_availability": {
    "cpu_cores": 16,
    "memory_gb": 64,
    "storage_tb": 2.5,
    "gpu_acceleration": true
  },
  "integration_status": {
    "mcp_protocol": "active",
    "agent_coordination": "enabled",
    "learning_engine": "operational"
  }
}
```

**Tool**: [`get_config_summary`](safla/mcp_stdio_server.py:1)
**Purpose**: Access configuration settings for analysis parameter optimization

```bash
# Get configuration summary for analysis optimization
use_mcp_tool safla get_config_summary '{}'

# Expected Response:
{
  "analysis_configuration": {
    "quality_thresholds": {
      "code_coverage": 95,
      "complexity_limit": 10,
      "maintainability_index": 80,
      "security_score": 90
    },
    "performance_baselines": {
      "response_time_ms": 100,
      "throughput_rps": 1000,
      "memory_usage_mb": 512,
      "cpu_utilization": 70
    },
    "analysis_depth": "comprehensive",
    "reporting_format": "detailed"
  },
  "integration_settings": {
    "real_time_monitoring": true,
    "automated_alerts": true,
    "learning_integration": true,
    "agent_collaboration": true
  }
}
```

### 2. Performance Analysis and Benchmarking Tools

#### Performance Bottleneck Analysis

**Tool**: [`analyze_performance_bottlenecks`](safla/mcp_stdio_server.py:1)
**Purpose**: Identify and analyze system performance bottlenecks

```bash
# Comprehensive performance bottleneck analysis
use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": 300,
  "include_memory_profile": true
}'

# Expected Response:
{
  "analysis_summary": {
    "total_bottlenecks": 7,
    "critical_issues": 2,
    "moderate_issues": 3,
    "minor_issues": 2,
    "overall_performance_score": 78.4
  },
  "critical_bottlenecks": [
    {
      "component": "database_query_optimization",
      "impact": "high",
      "severity": "critical",
      "description": "Slow JOIN operations on user_data table",
      "performance_impact": "45% response time increase",
      "recommendation": "Add composite index on (user_id, created_at)",
      "estimated_improvement": "60% faster queries"
    },
    {
      "component": "memory_allocation",
      "impact": "high", 
      "severity": "critical",
      "description": "Excessive object creation in request processing",
      "performance_impact": "30% memory overhead",
      "recommendation": "Implement object pooling for request handlers",
      "estimated_improvement": "40% memory reduction"
    }
  ],
  "memory_profile": {
    "peak_usage": "2.1GB",
    "average_usage": "1.8GB",
    "gc_frequency": "every_2.3s",
    "memory_leaks": "none_detected",
    "optimization_opportunities": [
      "reduce_object_churn",
      "optimize_cache_size",
      "implement_lazy_loading"
    ]
  },
  "recommendations": {
    "immediate_actions": [
      "Optimize database queries",
      "Implement memory pooling"
    ],
    "medium_term": [
      "Refactor request processing pipeline",
      "Implement caching strategy"
    ],
    "long_term": [
      "Consider microservices architecture",
      "Implement horizontal scaling"
    ]
  }
}
```

#### Memory Performance Analysis

**Tool**: [`benchmark_memory_performance`](safla/mcp_stdio_server.py:1)
**Purpose**: Comprehensive memory subsystem performance analysis

```bash
# Detailed memory performance benchmarking
use_mcp_tool safla benchmark_memory_performance '{
  "test_duration": 180,
  "memory_patterns": ["sequential_access", "random_access", "bulk_operations", "concurrent_access"]
}'

# Expected Response:
{
  "benchmark_results": {
    "overall_score": 87.3,
    "memory_efficiency": 91.2,
    "access_performance": 84.7,
    "allocation_efficiency": 89.1
  },
  "pattern_analysis": {
    "sequential_access": {
      "throughput_mbps": 2847,
      "latency_ns": 45,
      "efficiency_score": 94.2,
      "bottlenecks": "none"
    },
    "random_access": {
      "throughput_mbps": 1234,
      "latency_ns": 127,
      "efficiency_score": 78.9,
      "bottlenecks": ["cache_misses", "memory_fragmentation"]
    },
    "bulk_operations": {
      "throughput_mbps": 3421,
      "latency_ns": 23,
      "efficiency_score": 96.7,
      "bottlenecks": "none"
    },
    "concurrent_access": {
      "throughput_mbps": 1876,
      "latency_ns": 89,
      "efficiency_score": 82.4,
      "bottlenecks": ["lock_contention", "false_sharing"]
    }
  },
  "optimization_recommendations": [
    {
      "area": "random_access_optimization",
      "priority": "high",
      "description": "Implement memory prefetching for random access patterns",
      "expected_improvement": "25% latency reduction"
    },
    {
      "area": "concurrent_access_optimization", 
      "priority": "medium",
      "description": "Reduce lock contention through lock-free data structures",
      "expected_improvement": "15% throughput increase"
    }
  ]
}
```

#### Vector Operations Performance Analysis

**Tool**: [`benchmark_vector_operations`](safla/mcp_stdio_server.py:1)
**Purpose**: Analyze vector memory operations performance for ML/AI workloads

```bash
# Vector operations performance benchmarking
use_mcp_tool safla benchmark_vector_operations '{
  "vector_count": 10000,
  "vector_dimensions": 512,
  "operations": ["similarity", "clustering", "indexing", "search"]
}'

# Expected Response:
{
  "benchmark_summary": {
    "overall_performance": 92.1,
    "gpu_acceleration": "enabled",
    "memory_efficiency": 89.7,
    "operation_throughput": 94.3
  },
  "operation_results": {
    "similarity": {
      "operations_per_second": 15420,
      "avg_latency_ms": 0.65,
      "memory_usage_mb": 234,
      "accuracy_score": 99.2,
      "optimization_level": "excellent"
    },
    "clustering": {
      "operations_per_second": 8760,
      "avg_latency_ms": 1.14,
      "memory_usage_mb": 456,
      "accuracy_score": 97.8,
      "optimization_level": "good"
    },
    "indexing": {
      "operations_per_second": 23450,
      "avg_latency_ms": 0.43,
      "memory_usage_mb": 189,
      "accuracy_score": 99.9,
      "optimization_level": "excellent"
    },
    "search": {
      "operations_per_second": 18920,
      "avg_latency_ms": 0.53,
      "memory_usage_mb": 167,
      "accuracy_score": 98.7,
      "optimization_level": "excellent"
    }
  },
  "performance_analysis": {
    "strengths": [
      "Excellent GPU utilization",
      "Optimal memory access patterns",
      "High accuracy maintenance"
    ],
    "improvement_areas": [
      "Clustering algorithm optimization",
      "Memory usage reduction for clustering"
    ],
    "recommendations": [
      "Implement hierarchical clustering for better memory efficiency",
      "Consider approximate algorithms for non-critical clustering tasks"
    ]
  }
}
```

### 3. Testing and Quality Analysis Tools

#### Integration Testing Analysis

**Tool**: [`run_integration_tests`](safla/mcp_stdio_server.py:1)
**Purpose**: Execute and analyze comprehensive integration testing

```bash
# Run comprehensive integration tests for analysis
use_mcp_tool safla run_integration_tests '{
  "test_suite": "comprehensive",
  "parallel": true,
  "verbose": true
}'

# Expected Response:
{
  "test_execution_summary": {
    "total_tests": 247,
    "passed": 239,
    "failed": 8,
    "skipped": 0,
    "execution_time": "8m 34s",
    "parallel_efficiency": 87.3
  },
  "test_categories": {
    "api_integration": {
      "total": 89,
      "passed": 87,
      "failed": 2,
      "coverage": 94.7
    },
    "database_integration": {
      "total": 67,
      "passed": 64,
      "failed": 3,
      "coverage": 91.2
    },
    "service_integration": {
      "total": 91,
      "passed": 88,
      "failed": 3,
      "coverage": 96.1
    }
  },
  "failure_analysis": [
    {
      "test_name": "api_rate_limiting_integration",
      "category": "api_integration",
      "failure_type": "timeout",
      "error_message": "Request timeout after 5000ms",
      "impact": "medium",
      "recommendation": "Increase timeout or optimize rate limiting logic"
    },
    {
      "test_name": "database_connection_pool",
      "category": "database_integration", 
      "failure_type": "connection_error",
      "error_message": "Connection pool exhausted",
      "impact": "high",
      "recommendation": "Increase pool size or implement connection recycling"
    }
  ],
  "quality_metrics": {
    "test_reliability": 96.8,
    "execution_stability": 94.2,
    "coverage_completeness": 93.7,
    "performance_impact": "minimal"
  }
}
```

#### Memory Operations Validation

**Tool**: [`validate_memory_operations`](safla/mcp_stdio_server.py:1)
**Purpose**: Validate memory operations and data integrity

```bash
# Comprehensive memory operations validation
use_mcp_tool safla validate_memory_operations '{
  "test_data_size": 100,
  "include_stress_test": true
}'

# Expected Response:
{
  "validation_summary": {
    "overall_status": "passed",
    "memory_integrity": "verified",
    "data_consistency": "maintained",
    "performance_impact": "acceptable"
  },
  "operation_validation": {
    "allocation_deallocation": {
      "status": "passed",
      "cycles_tested": 10000,
      "memory_leaks": "none",
      "fragmentation_level": "low"
    },
    "data_integrity": {
      "status": "passed",
      "corruption_checks": "passed",
      "checksum_validation": "passed",
      "boundary_checks": "passed"
    },
    "concurrent_access": {
      "status": "passed",
      "race_conditions": "none",
      "deadlock_detection": "none",
      "data_races": "none"
    }
  },
  "stress_test_results": {
    "peak_memory_usage": "1.8GB",
    "sustained_load_duration": "15m",
    "performance_degradation": "2.3%",
    "recovery_time": "1.2s",
    "stability_score": 97.8
  },
  "recommendations": [
    "Memory operations are performing optimally",
    "Consider implementing memory compaction for long-running processes",
    "Monitor memory fragmentation in production environments"
  ]
}
```

### 4. Agent Coordination and Analysis Tools

#### Agent Session Analysis

**Tool**: [`create_agent_session`](safla/mcp_stdio_server.py:1)
**Purpose**: Create specialized analysis agents for focused critique

```bash
# Create cognitive agent for code quality analysis
use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "code_quality_analysis",
    "optimization_level": "comprehensive",
    "learning_enabled": true,
    "analysis_depth": "deep"
  },
  "timeout_seconds": 7200
}'

# Expected Response:
{
  "session_id": "quality_analysis_agent_001",
  "agent_type": "cognitive",
  "status": "active",
  "capabilities": [
    "static_code_analysis",
    "architectural_review",
    "security_assessment",
    "performance_analysis",
    "best_practices_validation"
  ],
  "configuration": {
    "analysis_frameworks": ["sonarqube", "eslint", "security_scanner"],
    "quality_thresholds": "enterprise_level",
    "reporting_detail": "comprehensive"
  }
}
```

**Tool**: [`interact_with_agent`](safla/mcp_stdio_server.py:1)
**Purpose**: Collaborate with analysis agents for specialized critique

```bash
# Interact with quality analysis agent for code review
use_mcp_tool safla interact_with_agent '{
  "session_id": "quality_analysis_agent_001",
  "command": "analyze_codebase",
  "parameters": {
    "target_directory": "src/",
    "analysis_types": ["complexity", "security", "maintainability", "performance"],
    "include_recommendations": true,
    "severity_threshold": "medium"
  }
}'

# Expected Response:
{
  "analysis_results": {
    "overall_quality_score": 84.7,
    "total_issues": 23,
    "critical_issues": 2,
    "major_issues": 7,
    "minor_issues": 14
  },
  "complexity_analysis": {
    "average_complexity": 6.8,
    "high_complexity_functions": 5,
    "complexity_hotspots": [
      {
        "file": "src/data/processor.ts",
        "function": "processComplexData",
        "complexity": 15,
        "recommendation": "Extract helper functions to reduce complexity"
      }
    ]
  },
  "security_analysis": {
    "vulnerabilities_found": 3,
    "security_score": 87.2,
    "critical_vulnerabilities": [
      {
        "type": "sql_injection",
        "location": "src/db/queries.ts:45",
        "severity": "high",
        "recommendation": "Use parameterized queries"
      }
    ]
  },
  "maintainability_analysis": {
    "maintainability_index": 82.1,
    "code_duplication": "7.3%",
    "documentation_coverage": "78.9%",
    "recommendations": [
      "Reduce code duplication in utility functions",
      "Improve documentation for public APIs"
    ]
  }
}
```

### 5. Meta-Cognitive Analysis Tools

#### System Awareness Analysis

**Tool**: [`get_system_awareness`](safla/mcp_stdio_server.py:1)
**Purpose**: Analyze current system self-awareness state for critique context

```bash
# Get system awareness for analysis context
use_mcp_tool safla get_system_awareness '{}'

# Expected Response:
{
  "awareness_state": {
    "current_focus": "quality_analysis",
    "attention_level": 0.92,
    "cognitive_load": 0.67,
    "analysis_confidence": 0.89
  },
  "system_understanding": {
    "architecture_comprehension": 0.94,
    "performance_awareness": 0.91,
    "security_understanding": 0.87,
    "quality_perception": 0.93
  },
  "meta_cognitive_insights": {
    "analysis_effectiveness": 0.88,
    "learning_rate": 0.15,
    "adaptation_speed": 0.82,
    "knowledge_integration": 0.91
  },
  "contextual_awareness": {
    "project_phase": "implementation",
    "quality_requirements": "high",
    "performance_constraints": "moderate",
    "security_requirements": "strict"
  }
}
```

#### Learning Metrics Analysis

**Tool**: [`get_learning_metrics`](safla/mcp_stdio_server.py:1)
**Purpose**: Analyze learning effectiveness and adaptation patterns

```bash
# Get comprehensive learning metrics for analysis improvement
use_mcp_tool safla get_learning_metrics '{
  "metric_type": "all",
  "time_range_hours": 168
}'

# Expected Response:
{
  "analysis_learning_metrics": {
    "pattern_recognition_accuracy": 92.4,
    "issue_prediction_success": 87.9,
    "recommendation_effectiveness": 89.7,
    "false_positive_reduction": 15.3
  },
  "adaptation_metrics": {
    "successful_adaptations": 34,
    "failed_adaptations": 3,
    "adaptation_success_rate": 91.9,
    "learning_velocity": 12.7
  },
  "knowledge_metrics": {
    "knowledge_retention": 94.8,
    "knowledge_application": 88.2,
    "cross_domain_transfer": 76.4,
    "expertise_development": 83.1
  },
  "improvement_trends": {
    "7_day_improvement": 8.3,
    "30_day_improvement": 23.7,
    "quality_trend": "improving",
    "efficiency_trend": "stable"
  }
}
```

#### Adaptation Pattern Analysis

**Tool**: [`analyze_adaptation_patterns`](safla/mcp_stdio_server.py:1)
**Purpose**: Analyze system adaptation trends for critique evolution

```bash
# Analyze adaptation patterns for critique improvement
use_mcp_tool safla analyze_adaptation_patterns '{
  "pattern_type": "all",
  "analysis_depth": "comprehensive",
  "time_window_days": 30
}'

# Expected Response:
{
  "analysis_adaptation_patterns": {
    "critique_methodology_evolution": {
      "pattern_type": "incremental_improvement",
      "confidence": 93.7,
      "trend": "improving",
      "adaptations": [
        "enhanced_context_awareness",
        "improved_recommendation_specificity",
        "better_priority_classification"
      ]
    },
    "quality_assessment_refinement": {
      "pattern_type": "targeted_optimization",
      "confidence": 89.2,
      "trend": "stable",
      "adaptations": [
        "refined_quality_thresholds",
        "improved_metric_weighting",
        "enhanced_trend_analysis"
      ]
    }
  },
  "behavioral_evolution": {
    "analysis_depth_adaptation": 0.87,
    "feedback_integration_improvement": 0.92,
    "recommendation_quality_enhancement": 0.89
  },
  "effectiveness_patterns": [
    {
      "pattern": "contextual_analysis_enhancement",
      "effectiveness": 94.6,
      "usage_frequency": "high",
      "impact": "significant"
    },
    {
      "pattern": "predictive_issue_identification",
      "effectiveness": 88.9,
      "usage_frequency": "medium",
      "impact": "moderate"
    }
  ]
}
```

### 6. Goal and Strategy Analysis Tools

#### Goal Progress Evaluation

**Tool**: [`evaluate_goal_progress`](safla/mcp_stdio_server.py:1)
**Purpose**: Evaluate progress toward quality and analysis goals

```bash
# Evaluate quality analysis goal progress
use_mcp_tool safla evaluate_goal_progress '{
  "goal_id": "comprehensive_quality_analysis",
  "include_recommendations": true
}'

# Expected Response:
{
  "goal_evaluation": {
    "goal_name": "Comprehensive Quality Analysis",
    "current_progress": 0.87,
    "target_achievement": {
      "code_quality_score": "achieved",
      "security_compliance": "in_progress",
      "performance_optimization": "exceeded",
      "documentation_coverage": "achieved"
    },
    "milestone_status": {
      "static_analysis_completion": "completed",
      "security_assessment": "in_progress",
      "performance_benchmarking": "completed",
      "integration_testing": "completed"
    }
  },
  "quality_metrics": {
    "overall_quality_improvement": 23.4,
    "critical_issues_resolved": 15,
    "performance_optimization_achieved": 18.7,
    "security_vulnerabilities_addressed": 8
  },
  "recommendations": [
    "Complete security assessment to achieve full compliance",
    "Document identified performance optimizations",
    "Establish continuous monitoring for quality metrics",
    "Create quality improvement roadmap for next iteration"
  ]
}
```

#### Strategy Effectiveness Analysis

**Tool**: [`evaluate_strategy_performance`](safla/mcp_stdio_server.py:1)
**Purpose**: Assess effectiveness of analysis and critique strategies

```bash
# Evaluate analysis strategy performance
use_mcp_tool safla evaluate_strategy_performance '{
  "strategy_id": "comprehensive_quality_analysis_001",
  "evaluation_period_hours": 168,
  "metrics": ["accuracy", "efficiency", "impact", "satisfaction"]
}'

# Expected Response:
{
  "strategy_performance": {
    "overall_effectiveness": 91.3,
    "accuracy_score": 94.7,
    "efficiency_rating": 87.9,
    "impact_measurement": 92.1,
    "stakeholder_satisfaction": 89.6
  },
  "performance_breakdown": {
    "issue_identification": {
      "accuracy": 96.2,
      "completeness": 89.7,
      "timeliness": 92.4
    },
    "recommendation_quality": {
      "actionability": 91.8,
      "specificity": 88.9,
      "implementation_success": 87.3
    },
    "analysis_efficiency": {
      "time_to_completion": 94.1,
      "resource_utilization": 81.7,
      "automation_level": 89.3
    }
  },
  "improvement_opportunities": [
    "Enhance automated analysis coverage",
    "Improve recommendation prioritization",
    "Increase stakeholder engagement in feedback process",
    "Develop predictive analysis capabilities"
  ],
  "success_factors": [
    "Comprehensive multi-dimensional analysis",
    "Integration with real-time monitoring",
    "Effective use of learning algorithms",
    "Strong collaboration with development teams"
  ]
}
```

This comprehensive MCP tools documentation provides detailed guidance for leveraging SAFLA's capabilities within the Critic mode, ensuring thorough analysis, evaluation, and quality assessment through intelligent tool utilization and data-driven insights.