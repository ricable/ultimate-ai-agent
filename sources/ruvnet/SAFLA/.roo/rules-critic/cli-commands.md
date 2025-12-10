# Critic Mode CLI Commands Documentation

## Overview

This document provides comprehensive CLI command guidance for the Critic mode within the SAFLA-aiGI integrated system. The Critic mode utilizes SAFLA's command-line interface to perform thorough analysis, evaluation, and quality assessment through direct system interaction, automated analysis workflows, and real-time monitoring capabilities.

## Core Analysis Command Workflows

### 1. System Health and Configuration Analysis

#### System Validation Commands
```bash
# Comprehensive system validation for analysis readiness
$ use_mcp_tool safla validate_installation '{}'
$ use_mcp_tool safla get_system_info '{}'
$ use_mcp_tool safla get_config_summary '{}'

# Monitor system health during analysis
$ use_mcp_tool safla monitor_system_health '{
  "check_interval": 30,
  "alert_thresholds": {
    "memory_usage": 0.85,
    "cpu_usage": 0.80,
    "disk_usage": 0.90
  }
}'

# Check GPU status for performance analysis
$ use_mcp_tool safla check_gpu_status '{}'

# Validate deployment status for analysis context
$ use_mcp_tool safla check_deployment_status '{"instance_name": "production"}'
```

#### Configuration Analysis Workflow
```bash
#!/bin/bash
# Comprehensive configuration analysis script

echo "=== SAFLA Configuration Analysis ==="

# Get system configuration
echo "1. System Configuration:"
use_mcp_tool safla get_config_summary '{}' | jq '.analysis_configuration'

# Validate installation integrity
echo "2. Installation Validation:"
use_mcp_tool safla validate_installation '{}' | jq '.validation_status'

# Check system information
echo "3. System Information:"
use_mcp_tool safla get_system_info '{}' | jq '.analysis_capabilities'

# Monitor current system health
echo "4. System Health Status:"
use_mcp_tool safla monitor_system_health '{
  "check_interval": 10,
  "alert_thresholds": {"memory_usage": 0.8, "cpu_usage": 0.75}
}' | jq '.current_status'

echo "Configuration analysis complete."
```

### 2. Performance Analysis and Benchmarking Commands

#### Performance Bottleneck Analysis
```bash
# Comprehensive performance bottleneck analysis
$ use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": 300,
  "include_memory_profile": true
}'

# Memory performance benchmarking
$ use_mcp_tool safla benchmark_memory_performance '{
  "test_duration": 180,
  "memory_patterns": ["sequential_access", "random_access", "bulk_operations", "concurrent_access"]
}'

# Vector operations performance analysis
$ use_mcp_tool safla benchmark_vector_operations '{
  "vector_count": 10000,
  "vector_dimensions": 512,
  "operations": ["similarity", "clustering", "indexing", "search"]
}'

# MCP protocol throughput analysis
$ use_mcp_tool safla benchmark_mcp_throughput '{
  "request_count": 500,
  "concurrent_connections": 8,
  "payload_size": "large"
}'
```

#### Automated Performance Analysis Workflow
```bash
#!/bin/bash
# Automated performance analysis and critique workflow

ANALYSIS_DURATION=300
REPORT_FILE="performance_analysis_$(date +%Y%m%d_%H%M%S).json"

echo "Starting comprehensive performance analysis..."

# 1. Analyze performance bottlenecks
echo "Analyzing performance bottlenecks..."
bottlenecks=$(use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": '$ANALYSIS_DURATION',
  "include_memory_profile": true
}')

# 2. Benchmark memory performance
echo "Benchmarking memory performance..."
memory_perf=$(use_mcp_tool safla benchmark_memory_performance '{
  "test_duration": 120,
  "memory_patterns": ["sequential_access", "random_access", "concurrent_access"]
}')

# 3. Benchmark vector operations
echo "Benchmarking vector operations..."
vector_perf=$(use_mcp_tool safla benchmark_vector_operations '{
  "vector_count": 5000,
  "vector_dimensions": 256,
  "operations": ["similarity", "search"]
}')

# 4. Compile comprehensive analysis report
echo "Compiling analysis report..."
cat > "$REPORT_FILE" << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "analysis_duration": $ANALYSIS_DURATION,
  "performance_bottlenecks": $bottlenecks,
  "memory_performance": $memory_perf,
  "vector_performance": $vector_perf,
  "analysis_summary": {
    "critical_issues": $(echo "$bottlenecks" | jq '.analysis_summary.critical_issues'),
    "overall_performance_score": $(echo "$bottlenecks" | jq '.analysis_summary.overall_performance_score'),
    "memory_efficiency": $(echo "$memory_perf" | jq '.benchmark_results.memory_efficiency'),
    "vector_performance": $(echo "$vector_perf" | jq '.benchmark_summary.overall_performance')
  }
}
EOF

echo "Performance analysis complete. Report saved to: $REPORT_FILE"

# 5. Generate recommendations
echo "Generating performance recommendations..."
critical_count=$(echo "$bottlenecks" | jq '.analysis_summary.critical_issues')
if [ "$critical_count" -gt 0 ]; then
  echo "CRITICAL: $critical_count critical performance issues identified"
  echo "$bottlenecks" | jq '.critical_bottlenecks[].recommendation'
fi

memory_score=$(echo "$memory_perf" | jq '.benchmark_results.memory_efficiency')
if (( $(echo "$memory_score < 85" | bc -l) )); then
  echo "WARNING: Memory efficiency below threshold ($memory_score%)"
  echo "$memory_perf" | jq '.optimization_recommendations[].description'
fi
```

### 3. Quality Analysis and Testing Commands

#### Comprehensive Testing Analysis
```bash
# Run comprehensive integration tests for analysis
$ use_mcp_tool safla run_integration_tests '{
  "test_suite": "comprehensive",
  "parallel": true,
  "verbose": true
}'

# Validate memory operations integrity
$ use_mcp_tool safla validate_memory_operations '{
  "test_data_size": 100,
  "include_stress_test": true
}'

# Test MCP connectivity and protocol compliance
$ use_mcp_tool safla test_mcp_connectivity '{
  "target_server": "safla",
  "test_depth": "comprehensive"
}'
```

#### Quality Analysis Automation Script
```bash
#!/bin/bash
# Comprehensive quality analysis automation

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
QUALITY_REPORT="quality_analysis_$TIMESTAMP.json"

echo "=== Comprehensive Quality Analysis ==="

# 1. Run integration tests
echo "Running integration tests..."
integration_results=$(use_mcp_tool safla run_integration_tests '{
  "test_suite": "comprehensive",
  "parallel": true,
  "verbose": false
}')

# 2. Validate memory operations
echo "Validating memory operations..."
memory_validation=$(use_mcp_tool safla validate_memory_operations '{
  "test_data_size": 50,
  "include_stress_test": true
}')

# 3. Test MCP connectivity
echo "Testing MCP connectivity..."
mcp_test=$(use_mcp_tool safla test_mcp_connectivity '{
  "target_server": "safla",
  "test_depth": "comprehensive"
}')

# 4. Calculate quality scores
total_tests=$(echo "$integration_results" | jq '.test_execution_summary.total_tests')
passed_tests=$(echo "$integration_results" | jq '.test_execution_summary.passed')
test_success_rate=$(echo "scale=2; $passed_tests * 100 / $total_tests" | bc)

memory_status=$(echo "$memory_validation" | jq -r '.validation_summary.overall_status')
mcp_status=$(echo "$mcp_test" | jq -r '.connectivity_status')

# 5. Generate quality report
cat > "$QUALITY_REPORT" << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "quality_summary": {
    "test_success_rate": $test_success_rate,
    "memory_validation": "$memory_status",
    "mcp_connectivity": "$mcp_status",
    "overall_quality_score": $(echo "scale=1; ($test_success_rate + 85 + 90) / 3" | bc)
  },
  "detailed_results": {
    "integration_tests": $integration_results,
    "memory_validation": $memory_validation,
    "mcp_connectivity": $mcp_test
  }
}
EOF

echo "Quality analysis complete. Report: $QUALITY_REPORT"

# 6. Quality assessment and recommendations
if (( $(echo "$test_success_rate < 95" | bc -l) )); then
  echo "WARNING: Test success rate below 95% ($test_success_rate%)"
  echo "Failed tests require immediate attention:"
  echo "$integration_results" | jq '.failure_analysis[].recommendation'
fi

if [ "$memory_status" != "passed" ]; then
  echo "CRITICAL: Memory validation failed"
  echo "$memory_validation" | jq '.recommendations[]'
fi
```

### 4. Agent Coordination and Analysis Commands

#### Agent-Based Analysis Workflow
```bash
# Create specialized analysis agent
$ analysis_agent=$(use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "comprehensive_analysis",
    "optimization_level": "aggressive",
    "learning_enabled": true
  },
  "timeout_seconds": 7200
}' | jq -r '.session_id')

# Interact with analysis agent for code review
$ use_mcp_tool safla interact_with_agent '{
  "session_id": "'$analysis_agent'",
  "command": "analyze_codebase",
  "parameters": {
    "target_directory": "src/",
    "analysis_types": ["complexity", "security", "maintainability", "performance"],
    "include_recommendations": true
  }
}'

# Get agent session status
$ use_mcp_tool safla list_agent_sessions '{
  "filter_by_type": "cognitive",
  "include_inactive": false
}'

# Terminate analysis agent when complete
$ use_mcp_tool safla terminate_agent_session '{
  "session_id": "'$analysis_agent'",
  "force": false
}'
```

#### Multi-Agent Analysis Coordination
```bash
#!/bin/bash
# Multi-agent analysis coordination script

declare -a agent_sessions=()
declare -a analysis_types=("code_quality" "security_analysis" "performance_review" "architecture_assessment")

echo "=== Multi-Agent Analysis Coordination ==="

# 1. Create specialized agents for different analysis types
for analysis_type in "${analysis_types[@]}"; do
  echo "Creating $analysis_type agent..."
  session_id=$(use_mcp_tool safla create_agent_session '{
    "agent_type": "cognitive",
    "session_config": {
      "focus": "'$analysis_type'",
      "optimization_level": "comprehensive",
      "learning_enabled": true
    },
    "timeout_seconds": 3600
  }' | jq -r '.session_id')
  
  agent_sessions+=("$session_id")
  echo "Created agent: $session_id for $analysis_type"
done

# 2. Execute parallel analysis with each agent
echo "Executing parallel analysis..."
for i in "${!agent_sessions[@]}"; do
  session_id="${agent_sessions[$i]}"
  analysis_type="${analysis_types[$i]}"
  
  echo "Starting $analysis_type analysis with agent $session_id"
  
  # Execute analysis in background
  (
    result=$(use_mcp_tool safla interact_with_agent '{
      "session_id": "'$session_id'",
      "command": "perform_analysis",
      "parameters": {
        "analysis_focus": "'$analysis_type'",
        "target_scope": "full_project",
        "detail_level": "comprehensive"
      }
    }')
    
    echo "$result" > "analysis_${analysis_type}_${session_id}.json"
    echo "Completed $analysis_type analysis"
  ) &
done

# 3. Wait for all analyses to complete
wait
echo "All parallel analyses completed"

# 4. Aggregate results
echo "Aggregating analysis results..."
aggregate_report="comprehensive_analysis_$(date +%Y%m%d_%H%M%S).json"

cat > "$aggregate_report" << 'EOF'
{
  "analysis_timestamp": "$(date -Iseconds)",
  "analysis_summary": {
    "total_agents": ${#agent_sessions[@]},
    "analysis_types": [$(printf '"%s",' "${analysis_types[@]}" | sed 's/,$//')]
  },
  "detailed_results": {}
}
EOF

# Add individual analysis results
for i in "${!agent_sessions[@]}"; do
  analysis_type="${analysis_types[$i]}"
  session_id="${agent_sessions[$i]}"
  
  if [ -f "analysis_${analysis_type}_${session_id}.json" ]; then
    echo "Adding $analysis_type results to aggregate report"
    # Merge results into aggregate report (simplified)
    jq --argjson new_result "$(cat "analysis_${analysis_type}_${session_id}.json")" \
       '.detailed_results["'$analysis_type'"] = $new_result' \
       "$aggregate_report" > temp.json && mv temp.json "$aggregate_report"
  fi
done

# 5. Cleanup agent sessions
echo "Cleaning up agent sessions..."
for session_id in "${agent_sessions[@]}"; do
  use_mcp_tool safla terminate_agent_session '{
    "session_id": "'$session_id'",
    "force": false
  }' > /dev/null
  echo "Terminated agent: $session_id"
done

echo "Multi-agent analysis complete. Report: $aggregate_report"
```

### 5. Meta-Cognitive Analysis Commands

#### System Awareness and Learning Analysis
```bash
# Get current system awareness state
$ use_mcp_tool safla get_system_awareness '{}'

# Analyze learning metrics
$ use_mcp_tool safla get_learning_metrics '{
  "metric_type": "all",
  "time_range_hours": 168
}'

# Analyze adaptation patterns
$ use_mcp_tool safla analyze_adaptation_patterns '{
  "pattern_type": "all",
  "analysis_depth": "comprehensive",
  "time_window_days": 30
}'

# Update learning parameters based on analysis
$ use_mcp_tool safla update_learning_parameters '{
  "learning_rate": 0.12,
  "adaptation_threshold": 0.85,
  "memory_retention": 0.92,
  "exploration_factor": 0.18
}'

# Trigger learning cycle for analysis improvement
$ use_mcp_tool safla trigger_learning_cycle '{
  "learning_type": "meta",
  "data_sources": ["analysis_outcomes", "feedback_quality", "recommendation_effectiveness"],
  "focus_areas": ["critique_accuracy", "recommendation_quality"]
}'
```

#### Learning and Adaptation Analysis Workflow
```bash
#!/bin/bash
# Comprehensive learning and adaptation analysis

LEARNING_REPORT="learning_analysis_$(date +%Y%m%d_%H%M%S).json"

echo "=== Learning and Adaptation Analysis ==="

# 1. Get system awareness state
echo "Analyzing system awareness..."
awareness=$(use_mcp_tool safla get_system_awareness '{}')

# 2. Get learning metrics
echo "Gathering learning metrics..."
learning_metrics=$(use_mcp_tool safla get_learning_metrics '{
  "metric_type": "all",
  "time_range_hours": 168
}')

# 3. Analyze adaptation patterns
echo "Analyzing adaptation patterns..."
adaptation_patterns=$(use_mcp_tool safla analyze_adaptation_patterns '{
  "pattern_type": "all",
  "analysis_depth": "comprehensive",
  "time_window_days": 30
}')

# 4. Calculate learning effectiveness scores
awareness_level=$(echo "$awareness" | jq '.awareness_state.attention_level')
learning_rate=$(echo "$learning_metrics" | jq '.adaptation_metrics.learning_velocity')
adaptation_success=$(echo "$learning_metrics" | jq '.adaptation_metrics.adaptation_success_rate')

overall_learning_score=$(echo "scale=2; ($awareness_level * 100 + $learning_rate + $adaptation_success) / 3" | bc)

# 5. Generate learning analysis report
cat > "$LEARNING_REPORT" << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "learning_summary": {
    "awareness_level": $awareness_level,
    "learning_velocity": $learning_rate,
    "adaptation_success_rate": $adaptation_success,
    "overall_learning_score": $overall_learning_score
  },
  "detailed_analysis": {
    "system_awareness": $awareness,
    "learning_metrics": $learning_metrics,
    "adaptation_patterns": $adaptation_patterns
  }
}
EOF

echo "Learning analysis complete. Report: $LEARNING_REPORT"

# 6. Optimization recommendations
if (( $(echo "$overall_learning_score < 80" | bc -l) )); then
  echo "RECOMMENDATION: Learning effectiveness below optimal threshold"
  
  if (( $(echo "$awareness_level < 0.8" | bc -l) )); then
    echo "- Increase system awareness focus"
    use_mcp_tool safla update_awareness_state '{
      "awareness_level": 0.9,
      "focus_areas": ["analysis_quality", "recommendation_effectiveness"],
      "introspection_depth": "deep"
    }'
  fi
  
  if (( $(echo "$learning_rate < 10" | bc -l) )); then
    echo "- Optimize learning parameters"
    use_mcp_tool safla update_learning_parameters '{
      "learning_rate": 0.15,
      "adaptation_threshold": 0.8,
      "exploration_factor": 0.25
    }'
  fi
fi

# 7. Trigger learning cycle if needed
if (( $(echo "$adaptation_success < 85" | bc -l) )); then
  echo "Triggering learning cycle for improvement..."
  use_mcp_tool safla trigger_learning_cycle '{
    "learning_type": "reinforcement",
    "data_sources": ["analysis_feedback", "recommendation_outcomes"],
    "focus_areas": ["critique_effectiveness", "learning_optimization"]
  }'
fi
```

### 6. Goal and Strategy Analysis Commands

#### Goal Progress and Strategy Evaluation
```bash
# Evaluate goal progress
$ use_mcp_tool safla evaluate_goal_progress '{
  "goal_id": "comprehensive_quality_analysis",
  "include_recommendations": true
}'

# Evaluate strategy performance
$ use_mcp_tool safla evaluate_strategy_performance '{
  "strategy_id": "multi_dimensional_analysis_001",
  "evaluation_period_hours": 168,
  "metrics": ["accuracy", "efficiency", "impact", "satisfaction"]
}'

# List available strategies
$ use_mcp_tool safla list_strategies '{
  "context_filter": "quality_analysis",
  "effectiveness_threshold": 0.85
}'

# Select optimal strategy for current context
$ use_mcp_tool safla select_optimal_strategy '{
  "context": "comprehensive_code_review",
  "constraints": {"time_limit": "4_hours", "detail_level": "high"},
  "objectives": ["quality_improvement", "issue_identification", "recommendation_generation"]
}'
```

#### Comprehensive Analysis Workflow with Goal Tracking
```bash
#!/bin/bash
# Comprehensive analysis workflow with goal tracking

ANALYSIS_GOAL="comprehensive_system_analysis_$(date +%Y%m%d)"
WORKFLOW_REPORT="analysis_workflow_$(date +%Y%m%d_%H%M%S).json"

echo "=== Comprehensive Analysis Workflow ==="

# 1. Create analysis goal
echo "Creating analysis goal..."
goal_result=$(use_mcp_tool safla create_goal '{
  "goal_name": "'$ANALYSIS_GOAL'",
  "description": "Comprehensive system analysis including performance, quality, and security assessment",
  "priority": "high",
  "target_metrics": {
    "analysis_completeness": 0.95,
    "issue_identification_accuracy": 0.90,
    "recommendation_quality": 0.85
  },
  "deadline": '$(date -d "+1 day" +%s)'
}')

goal_id=$(echo "$goal_result" | jq -r '.goal_id')
echo "Created goal: $goal_id"

# 2. Select optimal analysis strategy
echo "Selecting optimal analysis strategy..."
strategy=$(use_mcp_tool safla select_optimal_strategy '{
  "context": "comprehensive_system_analysis",
  "constraints": {"time_limit": "6_hours", "resource_usage": "high"},
  "objectives": ["thorough_analysis", "actionable_recommendations", "quality_improvement"]
}')

strategy_id=$(echo "$strategy" | jq -r '.selected_strategy.strategy_id')
echo "Selected strategy: $strategy_id"

# 3. Execute comprehensive analysis
echo "Executing comprehensive analysis..."

# Performance analysis
performance_analysis=$(use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": 180,
  "include_memory_profile": true
}')

# Quality analysis
quality_analysis=$(use_mcp_tool safla run_integration_tests '{
  "test_suite": "comprehensive",
  "parallel": true,
  "verbose": false
}')

# Security analysis (via agent)
security_agent=$(use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {"focus": "security_analysis", "optimization_level": "comprehensive"},
  "timeout_seconds": 3600
}' | jq -r '.session_id')

security_analysis=$(use_mcp_tool safla interact_with_agent '{
  "session_id": "'$security_agent'",
  "command": "security_assessment",
  "parameters": {"scope": "full_system", "depth": "comprehensive"}
}')

# 4. Update goal progress
echo "Updating goal progress..."
use_mcp_tool safla update_goal '{
  "goal_id": "'$goal_id'",
  "progress": 0.8,
  "notes": "Comprehensive analysis 80% complete - performance, quality, and security analysis completed"
}'

# 5. Generate comprehensive report
echo "Generating comprehensive analysis report..."
cat > "$WORKFLOW_REPORT" << EOF
{
  "workflow_timestamp": "$(date -Iseconds)",
  "goal_tracking": {
    "goal_id": "$goal_id",
    "strategy_id": "$strategy_id",
    "progress": 0.8
  },
  "analysis_results": {
    "performance_analysis": $performance_analysis,
    "quality_analysis": $quality_analysis,
    "security_analysis": $security_analysis
  },
  "summary": {
    "critical_issues": $(echo "$performance_analysis" | jq '.analysis_summary.critical_issues'),
    "test_success_rate": $(echo "$quality_analysis" | jq '.test_execution_summary.passed / .test_execution_summary.total_tests * 100'),
    "security_score": $(echo "$security_analysis" | jq '.security_score // 85')
  }
}
EOF

# 6. Evaluate final goal progress
echo "Evaluating final goal progress..."
final_evaluation=$(use_mcp_tool safla evaluate_goal_progress '{
  "goal_id": "'$goal_id'",
  "include_recommendations": true
}')

# 7. Complete goal if targets met
progress=$(echo "$final_evaluation" | jq '.goal_evaluation.progress')
if (( $(echo "$progress >= 0.95" | bc -l) )); then
  use_mcp_tool safla update_goal '{
    "goal_id": "'$goal_id'",
    "status": "completed",
    "progress": 1.0,
    "notes": "Comprehensive analysis completed successfully"
  }'
  echo "Goal completed successfully!"
else
  echo "Goal progress: $progress - additional work needed"
fi

# 8. Cleanup
use_mcp_tool safla terminate_agent_session '{"session_id": "'$security_agent'"}'

echo "Comprehensive analysis workflow complete. Report: $WORKFLOW_REPORT"
```

This comprehensive CLI commands documentation provides detailed guidance for executing critic mode operations using SAFLA's command-line interface, ensuring thorough analysis, evaluation, and quality assessment through systematic command workflows and automation scripts.