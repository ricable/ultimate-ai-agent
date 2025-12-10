# Critic Mode Workflow Documentation

## Overview

This document defines the comprehensive workflow patterns for the Critic mode within the SAFLA-aiGI integrated system. The Critic mode orchestrates analytical and evaluative processes through systematic workflows that leverage SAFLA's MCP tools, CLI commands, and resource access patterns to ensure thorough analysis, quality assessment, and continuous improvement.

## Core Workflow Architecture

### 1. Analysis Initiation and Context Establishment

#### Workflow Entry Points
```bash
# Standard critic mode activation
new_task critic "Comprehensive analysis of system performance and code quality"

# Triggered from other modes for evaluation
new_task critic "Evaluate implementation quality from code mode output"

# Scheduled analysis workflow
new_task critic "Periodic system health and performance analysis"

# Emergency analysis workflow
new_task critic "Critical issue analysis and resolution recommendations"
```

#### Context Establishment Workflow
```bash
#!/bin/bash
# Context establishment and analysis preparation workflow

ANALYSIS_ID="analysis_$(date +%Y%m%d_%H%M%S)"
CONTEXT_FILE="context_${ANALYSIS_ID}.json"

echo "=== Critic Mode Context Establishment ==="

# 1. System validation and readiness check
echo "Step 1: System Validation"
system_status=$(use_mcp_tool safla validate_installation '{}')
system_ready=$(echo "$system_status" | jq -r '.validation_summary.overall_status')

if [ "$system_ready" != "passed" ]; then
  echo "CRITICAL: System not ready for analysis"
  echo "$system_status" | jq '.validation_issues[]'
  exit 1
fi

# 2. Gather system configuration and status
echo "Step 2: Configuration Analysis"
config_data=$(access_mcp_resource safla "safla://config")
status_data=$(access_mcp_resource safla "safla://status")

# 3. Establish performance baselines
echo "Step 3: Performance Baseline Establishment"
baseline_data=$(access_mcp_resource safla "safla://performance-baselines")
current_metrics=$(access_mcp_resource safla "safla://performance-metrics")

# 4. Initialize meta-cognitive awareness
echo "Step 4: Meta-Cognitive Initialization"
awareness_state=$(use_mcp_tool safla get_system_awareness '{}')
use_mcp_tool safla update_awareness_state '{
  "awareness_level": 0.9,
  "focus_areas": ["analysis_quality", "recommendation_effectiveness", "system_optimization"],
  "introspection_depth": "deep"
}'

# 5. Create analysis context
cat > "$CONTEXT_FILE" << EOF
{
  "analysis_id": "$ANALYSIS_ID",
  "initialization_timestamp": "$(date -Iseconds)",
  "context_establishment": {
    "system_status": $system_status,
    "configuration": $config_data,
    "current_status": $status_data,
    "performance_baseline": $baseline_data,
    "current_metrics": $current_metrics,
    "awareness_state": $awareness_state
  },
  "analysis_scope": {
    "performance_analysis": true,
    "quality_assessment": true,
    "security_evaluation": true,
    "optimization_recommendations": true
  }
}
EOF

echo "Context established. Analysis ID: $ANALYSIS_ID"
echo "Context file: $CONTEXT_FILE"
```

### 2. Multi-Dimensional Analysis Workflow

#### Comprehensive Analysis Orchestration
```bash
#!/bin/bash
# Multi-dimensional analysis orchestration workflow

source context_${ANALYSIS_ID}.json 2>/dev/null || {
  echo "ERROR: Analysis context not found"
  exit 1
}

ANALYSIS_REPORT="comprehensive_analysis_${ANALYSIS_ID}.json"

echo "=== Multi-Dimensional Analysis Orchestration ==="

# 1. Performance Analysis Dimension
echo "Dimension 1: Performance Analysis"
performance_analysis() {
  echo "  Analyzing performance bottlenecks..."
  bottlenecks=$(use_mcp_tool safla analyze_performance_bottlenecks '{
    "duration_seconds": 300,
    "include_memory_profile": true
  }')
  
  echo "  Benchmarking memory performance..."
  memory_bench=$(use_mcp_tool safla benchmark_memory_performance '{
    "test_duration": 120,
    "memory_patterns": ["sequential_access", "random_access", "concurrent_access"]
  }')
  
  echo "  Benchmarking vector operations..."
  vector_bench=$(use_mcp_tool safla benchmark_vector_operations '{
    "vector_count": 5000,
    "vector_dimensions": 512,
    "operations": ["similarity", "clustering", "search"]
  }')
  
  # Compile performance analysis results
  cat > "performance_analysis_${ANALYSIS_ID}.json" << EOF
{
  "analysis_type": "performance",
  "timestamp": "$(date -Iseconds)",
  "results": {
    "bottlenecks": $bottlenecks,
    "memory_benchmark": $memory_bench,
    "vector_benchmark": $vector_bench
  }
}
EOF
}

# 2. Quality Assessment Dimension
echo "Dimension 2: Quality Assessment"
quality_assessment() {
  echo "  Running integration tests..."
  test_results=$(use_mcp_tool safla run_integration_tests '{
    "test_suite": "comprehensive",
    "parallel": true,
    "verbose": false
  }')
  
  echo "  Validating memory operations..."
  memory_validation=$(use_mcp_tool safla validate_memory_operations '{
    "test_data_size": 50,
    "include_stress_test": true
  }')
  
  echo "  Testing MCP connectivity..."
  mcp_test=$(use_mcp_tool safla test_mcp_connectivity '{
    "target_server": "safla",
    "test_depth": "comprehensive"
  }')
  
  # Compile quality assessment results
  cat > "quality_assessment_${ANALYSIS_ID}.json" << EOF
{
  "analysis_type": "quality",
  "timestamp": "$(date -Iseconds)",
  "results": {
    "integration_tests": $test_results,
    "memory_validation": $memory_validation,
    "mcp_connectivity": $mcp_test
  }
}
EOF
}

# 3. Security Evaluation Dimension
echo "Dimension 3: Security Evaluation"
security_evaluation() {
  echo "  Creating security analysis agent..."
  security_agent=$(use_mcp_tool safla create_agent_session '{
    "agent_type": "cognitive",
    "session_config": {
      "focus": "security_analysis",
      "optimization_level": "comprehensive",
      "learning_enabled": true
    },
    "timeout_seconds": 3600
  }' | jq -r '.session_id')
  
  echo "  Performing security assessment..."
  security_analysis=$(use_mcp_tool safla interact_with_agent '{
    "session_id": "'$security_agent'",
    "command": "security_assessment",
    "parameters": {
      "scope": "full_system",
      "depth": "comprehensive",
      "include_recommendations": true
    }
  }')
  
  # Compile security evaluation results
  cat > "security_evaluation_${ANALYSIS_ID}.json" << EOF
{
  "analysis_type": "security",
  "timestamp": "$(date -Iseconds)",
  "agent_session": "$security_agent",
  "results": {
    "security_assessment": $security_analysis
  }
}
EOF
  
  # Cleanup security agent
  use_mcp_tool safla terminate_agent_session '{"session_id": "'$security_agent'"}'
}

# 4. Meta-Cognitive Analysis Dimension
echo "Dimension 4: Meta-Cognitive Analysis"
metacognitive_analysis() {
  echo "  Analyzing learning metrics..."
  learning_metrics=$(use_mcp_tool safla get_learning_metrics '{
    "metric_type": "all",
    "time_range_hours": 168
  }')
  
  echo "  Analyzing adaptation patterns..."
  adaptation_patterns=$(use_mcp_tool safla analyze_adaptation_patterns '{
    "pattern_type": "all",
    "analysis_depth": "comprehensive",
    "time_window_days": 30
  }')
  
  echo "  Evaluating goal progress..."
  goal_progress=$(use_mcp_tool safla evaluate_goal_progress '{
    "include_recommendations": true
  }')
  
  # Compile meta-cognitive analysis results
  cat > "metacognitive_analysis_${ANALYSIS_ID}.json" << EOF
{
  "analysis_type": "metacognitive",
  "timestamp": "$(date -Iseconds)",
  "results": {
    "learning_metrics": $learning_metrics,
    "adaptation_patterns": $adaptation_patterns,
    "goal_progress": $goal_progress
  }
}
EOF
}

# Execute all analysis dimensions in parallel
echo "Executing parallel analysis dimensions..."
performance_analysis &
quality_assessment &
security_evaluation &
metacognitive_analysis &

# Wait for all analyses to complete
wait
echo "All analysis dimensions completed"
```

### 3. Analysis Integration and Synthesis Workflow

#### Results Integration and Synthesis
```bash
#!/bin/bash
# Analysis integration and synthesis workflow

SYNTHESIS_REPORT="analysis_synthesis_${ANALYSIS_ID}.json"

echo "=== Analysis Integration and Synthesis ==="

# 1. Load all analysis results
echo "Loading analysis results..."
performance_results=$(cat "performance_analysis_${ANALYSIS_ID}.json" 2>/dev/null || echo '{}')
quality_results=$(cat "quality_assessment_${ANALYSIS_ID}.json" 2>/dev/null || echo '{}')
security_results=$(cat "security_evaluation_${ANALYSIS_ID}.json" 2>/dev/null || echo '{}')
metacognitive_results=$(cat "metacognitive_analysis_${ANALYSIS_ID}.json" 2>/dev/null || echo '{}')

# 2. Extract key metrics for synthesis
echo "Extracting key metrics..."

# Performance metrics
performance_score=$(echo "$performance_results" | jq '.results.bottlenecks.analysis_summary.overall_performance_score // 0')
memory_efficiency=$(echo "$performance_results" | jq '.results.memory_benchmark.benchmark_results.memory_efficiency // 0')
vector_performance=$(echo "$performance_results" | jq '.results.vector_benchmark.benchmark_summary.overall_performance // 0')

# Quality metrics
test_success_rate=$(echo "$quality_results" | jq '.results.integration_tests.test_execution_summary.passed / .results.integration_tests.test_execution_summary.total_tests * 100 // 0')
memory_validation_status=$(echo "$quality_results" | jq -r '.results.memory_validation.validation_summary.overall_status // "unknown"')
mcp_connectivity_score=$(echo "$quality_results" | jq '.results.mcp_connectivity.connectivity_score // 0')

# Security metrics
security_score=$(echo "$security_results" | jq '.results.security_assessment.security_score // 0')
vulnerability_count=$(echo "$security_results" | jq '.results.security_assessment.vulnerability_summary.total_vulnerabilities // 0')

# Meta-cognitive metrics
learning_velocity=$(echo "$metacognitive_results" | jq '.results.learning_metrics.learning_performance.velocity // 0')
adaptation_success=$(echo "$metacognitive_results" | jq '.results.adaptation_patterns.adaptation_metrics.success_rate // 0')

# 3. Calculate composite scores
echo "Calculating composite scores..."
overall_performance=$(echo "scale=1; ($performance_score + $memory_efficiency + $vector_performance) / 3" | bc)
overall_quality=$(echo "scale=1; ($test_success_rate + $mcp_connectivity_score) / 2" | bc)
overall_security=$(echo "scale=1; $security_score - ($vulnerability_count * 5)" | bc)
overall_metacognitive=$(echo "scale=1; ($learning_velocity + $adaptation_success) / 2" | bc)

# Calculate weighted overall score
overall_system_score=$(echo "scale=1; ($overall_performance * 0.3 + $overall_quality * 0.3 + $overall_security * 0.25 + $overall_metacognitive * 0.15)" | bc)

# 4. Generate synthesis report
cat > "$SYNTHESIS_REPORT" << EOF
{
  "analysis_id": "$ANALYSIS_ID",
  "synthesis_timestamp": "$(date -Iseconds)",
  "composite_scores": {
    "overall_performance": $overall_performance,
    "overall_quality": $overall_quality,
    "overall_security": $overall_security,
    "overall_metacognitive": $overall_metacognitive,
    "overall_system_score": $overall_system_score
  },
  "detailed_metrics": {
    "performance": {
      "performance_score": $performance_score,
      "memory_efficiency": $memory_efficiency,
      "vector_performance": $vector_performance
    },
    "quality": {
      "test_success_rate": $test_success_rate,
      "memory_validation": "$memory_validation_status",
      "mcp_connectivity": $mcp_connectivity_score
    },
    "security": {
      "security_score": $security_score,
      "vulnerability_count": $vulnerability_count
    },
    "metacognitive": {
      "learning_velocity": $learning_velocity,
      "adaptation_success": $adaptation_success
    }
  },
  "raw_analysis_results": {
    "performance": $performance_results,
    "quality": $quality_results,
    "security": $security_results,
    "metacognitive": $metacognitive_results
  }
}
EOF

echo "Analysis synthesis complete. Report: $SYNTHESIS_REPORT"
echo "Overall System Score: $overall_system_score"
```

### 4. Recommendation Generation and Prioritization Workflow

#### Intelligent Recommendation System
```bash
#!/bin/bash
# Recommendation generation and prioritization workflow

RECOMMENDATIONS_REPORT="recommendations_${ANALYSIS_ID}.json"

echo "=== Recommendation Generation and Prioritization ==="

# 1. Load synthesis results
synthesis_data=$(cat "$SYNTHESIS_REPORT" 2>/dev/null || {
  echo "ERROR: Synthesis report not found"
  exit 1
})

# 2. Extract scores for recommendation logic
overall_performance=$(echo "$synthesis_data" | jq '.composite_scores.overall_performance')
overall_quality=$(echo "$synthesis_data" | jq '.composite_scores.overall_quality')
overall_security=$(echo "$synthesis_data" | jq '.composite_scores.overall_security')
overall_system_score=$(echo "$synthesis_data" | jq '.composite_scores.overall_system_score')

# 3. Generate performance recommendations
echo "Generating performance recommendations..."
performance_recommendations=()

if (( $(echo "$overall_performance < 70" | bc -l) )); then
  performance_recommendations+=('{"priority": "critical", "category": "performance", "description": "System performance below acceptable threshold", "action": "immediate_optimization_required", "estimated_impact": "high"}')
fi

memory_efficiency=$(echo "$synthesis_data" | jq '.detailed_metrics.performance.memory_efficiency')
if (( $(echo "$memory_efficiency < 80" | bc -l) )); then
  performance_recommendations+=('{"priority": "high", "category": "memory", "description": "Memory efficiency optimization needed", "action": "optimize_memory_operations", "estimated_impact": "medium"}')
fi

# 4. Generate quality recommendations
echo "Generating quality recommendations..."
quality_recommendations=()

test_success_rate=$(echo "$synthesis_data" | jq '.detailed_metrics.quality.test_success_rate')
if (( $(echo "$test_success_rate < 95" | bc -l) )); then
  quality_recommendations+=('{"priority": "high", "category": "testing", "description": "Test success rate below target", "action": "investigate_test_failures", "estimated_impact": "medium"}')
fi

memory_validation=$(echo "$synthesis_data" | jq -r '.detailed_metrics.quality.memory_validation')
if [ "$memory_validation" != "passed" ]; then
  quality_recommendations+=('{"priority": "critical", "category": "memory_integrity", "description": "Memory validation failed", "action": "immediate_memory_system_review", "estimated_impact": "high"}')
fi

# 5. Generate security recommendations
echo "Generating security recommendations..."
security_recommendations=()

security_score=$(echo "$synthesis_data" | jq '.detailed_metrics.security.security_score')
if (( $(echo "$security_score < 80" | bc -l) )); then
  security_recommendations+=('{"priority": "high", "category": "security", "description": "Security posture needs improvement", "action": "comprehensive_security_review", "estimated_impact": "high"}')
fi

vulnerability_count=$(echo "$synthesis_data" | jq '.detailed_metrics.security.vulnerability_count')
if [ "$vulnerability_count" -gt 0 ]; then
  security_recommendations+=('{"priority": "critical", "category": "vulnerabilities", "description": "'$vulnerability_count' vulnerabilities identified", "action": "immediate_vulnerability_remediation", "estimated_impact": "critical"}')
fi

# 6. Generate meta-cognitive recommendations
echo "Generating meta-cognitive recommendations..."
metacognitive_recommendations=()

learning_velocity=$(echo "$synthesis_data" | jq '.detailed_metrics.metacognitive.learning_velocity')
if (( $(echo "$learning_velocity < 10" | bc -l) )); then
  metacognitive_recommendations+=('{"priority": "medium", "category": "learning", "description": "Learning velocity below optimal", "action": "optimize_learning_parameters", "estimated_impact": "medium"}')
fi

adaptation_success=$(echo "$synthesis_data" | jq '.detailed_metrics.metacognitive.adaptation_success')
if (( $(echo "$adaptation_success < 80" | bc -l) )); then
  metacognitive_recommendations+=('{"priority": "medium", "category": "adaptation", "description": "Adaptation success rate suboptimal", "action": "enhance_adaptation_mechanisms", "estimated_impact": "medium"}')
fi

# 7. Prioritize and consolidate recommendations
echo "Prioritizing recommendations..."

# Combine all recommendations
all_recommendations=$(printf '%s\n' "${performance_recommendations[@]}" "${quality_recommendations[@]}" "${security_recommendations[@]}" "${metacognitive_recommendations[@]}" | jq -s '.')

# Sort by priority (critical > high > medium > low)
prioritized_recommendations=$(echo "$all_recommendations" | jq 'sort_by(.priority == "critical" | not) | sort_by(.priority == "high" | not)')

# 8. Generate final recommendations report
cat > "$RECOMMENDATIONS_REPORT" << EOF
{
  "analysis_id": "$ANALYSIS_ID",
  "recommendations_timestamp": "$(date -Iseconds)",
  "recommendation_summary": {
    "total_recommendations": $(echo "$prioritized_recommendations" | jq 'length'),
    "critical_count": $(echo "$prioritized_recommendations" | jq '[.[] | select(.priority == "critical")] | length'),
    "high_count": $(echo "$prioritized_recommendations" | jq '[.[] | select(.priority == "high")] | length'),
    "medium_count": $(echo "$prioritized_recommendations" | jq '[.[] | select(.priority == "medium")] | length'),
    "overall_system_health": "$overall_system_score"
  },
  "prioritized_recommendations": $prioritized_recommendations,
  "implementation_roadmap": {
    "immediate_actions": $(echo "$prioritized_recommendations" | jq '[.[] | select(.priority == "critical")]'),
    "short_term_actions": $(echo "$prioritized_recommendations" | jq '[.[] | select(.priority == "high")]'),
    "medium_term_actions": $(echo "$prioritized_recommendations" | jq '[.[] | select(.priority == "medium")]')
  }
}
EOF

echo "Recommendations generated. Report: $RECOMMENDATIONS_REPORT"

# 9. Display critical recommendations
critical_count=$(echo "$prioritized_recommendations" | jq '[.[] | select(.priority == "critical")] | length')
if [ "$critical_count" -gt 0 ]; then
  echo ""
  echo "CRITICAL RECOMMENDATIONS REQUIRING IMMEDIATE ATTENTION:"
  echo "$prioritized_recommendations" | jq -r '.[] | select(.priority == "critical") | "- " + .description + " (" + .action + ")"'
fi
```

### 5. Workflow Completion and Handoff

#### Analysis Completion and Mode Coordination
```bash
#!/bin/bash
# Analysis completion and mode coordination workflow

FINAL_REPORT="critic_analysis_final_${ANALYSIS_ID}.json"

echo "=== Analysis Completion and Mode Coordination ==="

# 1. Consolidate all analysis artifacts
echo "Consolidating analysis artifacts..."
context_data=$(cat "context_${ANALYSIS_ID}.json" 2>/dev/null || echo '{}')
synthesis_data=$(cat "$SYNTHESIS_REPORT" 2>/dev/null || echo '{}')
recommendations_data=$(cat "$RECOMMENDATIONS_REPORT" 2>/dev/null || echo '{}')

# 2. Generate final comprehensive report
cat > "$FINAL_REPORT" << EOF
{
  "analysis_id": "$ANALYSIS_ID",
  "completion_timestamp": "$(date -Iseconds)",
  "analysis_summary": {
    "overall_system_score": $(echo "$synthesis_data" | jq '.composite_scores.overall_system_score'),
    "critical_issues": $(echo "$recommendations_data" | jq '.recommendation_summary.critical_count'),
    "total_recommendations": $(echo "$recommendations_data" | jq '.recommendation_summary.total_recommendations'),
    "analysis_duration": "$(date -d "$(echo "$context_data" | jq -r '.initialization_timestamp')" +%s | xargs -I {} echo "$(date +%s) - {}" | bc) seconds"
  },
  "complete_analysis": {
    "context": $context_data,
    "synthesis": $synthesis_data,
    "recommendations": $recommendations_data
  }
}
EOF

# 3. Determine next workflow actions
overall_score=$(echo "$synthesis_data" | jq '.composite_scores.overall_system_score')
critical_count=$(echo "$recommendations_data" | jq '.recommendation_summary.critical_count')

echo "Analysis Complete - Overall Score: $overall_score"

# 4. Workflow handoff logic
if [ "$critical_count" -gt 0 ]; then
  echo "CRITICAL ISSUES DETECTED - Spawning immediate resolution workflow"
  
  # Spawn code mode for critical fixes
  new_task code "Address critical issues identified in analysis $ANALYSIS_ID: $(echo "$recommendations_data" | jq -r '.implementation_roadmap.immediate_actions[].description' | head -3 | tr '\n' '; ')"
  
elif (( $(echo "$overall_score < 75" | bc -l) )); then
  echo "PERFORMANCE OPTIMIZATION NEEDED - Spawning optimization workflow"
  
  # Spawn orchestrator mode for comprehensive optimization
  new_task orchestrator "Orchestrate system optimization based on analysis $ANALYSIS_ID recommendations"
  
else
  echo "SYSTEM HEALTHY - Scheduling routine maintenance"
  
  # Spawn maintenance workflow
  new_task architect "Plan maintenance and improvement roadmap based on analysis $ANALYSIS_ID"
fi

# 5. Update meta-cognitive state with analysis results
echo "Updating meta-cognitive state..."
use_mcp_tool safla trigger_learning_cycle '{
  "learning_type": "meta",
  "data_sources": ["analysis_outcomes", "recommendation_effectiveness"],
  "focus_areas": ["analysis_quality", "recommendation_accuracy"]
}'

# 6. Archive analysis artifacts
echo "Archiving analysis artifacts..."
mkdir -p "analysis_archive/$ANALYSIS_ID"
mv *_${ANALYSIS_ID}.json "analysis_archive/$ANALYSIS_ID/"

echo "Critic mode analysis workflow complete."
echo "Final report: $FINAL_REPORT"
echo "Analysis artifacts archived in: analysis_archive/$ANALYSIS_ID/"

# 7. Workflow completion
attempt_completion "Comprehensive critic analysis completed for $ANALYSIS_ID. Overall system score: $overall_score. Generated $(echo "$recommendations_data" | jq '.recommendation_summary.total_recommendations') recommendations with $critical_count critical issues. Analysis artifacts archived and appropriate follow-up workflows initiated."
```

This comprehensive workflow documentation provides systematic guidance for executing critic mode operations through structured, automated workflows that leverage SAFLA's full capabilities for thorough analysis, evaluation, and continuous improvement within the aiGI framework.