# Scorer Mode Resources Documentation

## Overview

The Scorer mode leverages SAFLA's comprehensive resource system to access real-time data, historical metrics, and analytical insights for quantitative evaluation and scoring. This document details the resource access patterns, data integration workflows, and resource utilization strategies for effective scoring within the aiGI framework.

## Core SAFLA Resources for Scoring

### 1. Performance and Metrics Resources

#### safla://performance-metrics
**Purpose**: Real-time performance data for scoring calculations
**Data Structure**: Current system performance indicators and measurements
**Scoring Applications**: Performance scoring, trend analysis, baseline comparisons

```bash
# Access current performance metrics for scoring
PERF_DATA=$(access_mcp_resource safla safla://performance-metrics)

# Extract specific performance scores
CPU_UTILIZATION=$(echo $PERF_DATA | jq '.cpu_utilization_percent')
MEMORY_USAGE=$(echo $PERF_DATA | jq '.memory_usage_percent')
RESPONSE_TIME=$(echo $PERF_DATA | jq '.average_response_time_ms')
THROUGHPUT=$(echo $PERF_DATA | jq '.requests_per_second')

# Calculate composite performance score
PERFORMANCE_SCORE=$(echo "scale=2; (100 - $CPU_UTILIZATION) * 0.3 + (100 - $MEMORY_USAGE) * 0.3 + (1000 / $RESPONSE_TIME) * 0.2 + ($THROUGHPUT / 100) * 0.2" | bc)

echo "Current Performance Score: $PERFORMANCE_SCORE/100"
```

#### safla://benchmark-results
**Purpose**: Historical benchmark data for comparative scoring
**Data Structure**: Benchmark test results, performance baselines, comparison metrics
**Scoring Applications**: Performance improvement scoring, regression detection, optimization effectiveness

```bash
# Access benchmark results for comparative scoring
BENCHMARK_DATA=$(access_mcp_resource safla safla://benchmark-results)

# Extract benchmark scores for comparison
CURRENT_BENCHMARK=$(echo $BENCHMARK_DATA | jq '.latest_benchmark.overall_score')
BASELINE_BENCHMARK=$(echo $BENCHMARK_DATA | jq '.baseline_benchmark.overall_score')
BEST_BENCHMARK=$(echo $BENCHMARK_DATA | jq '.best_benchmark.overall_score')

# Calculate improvement scores
IMPROVEMENT_SCORE=$(echo "scale=2; ($CURRENT_BENCHMARK - $BASELINE_BENCHMARK) / $BASELINE_BENCHMARK * 100" | bc)
POTENTIAL_SCORE=$(echo "scale=2; ($BEST_BENCHMARK - $CURRENT_BENCHMARK) / $BEST_BENCHMARK * 100" | bc)

echo "Performance Improvement: $IMPROVEMENT_SCORE%"
echo "Optimization Potential: $POTENTIAL_SCORE%"
```

#### safla://performance-baselines
**Purpose**: Established performance baselines for scoring context
**Data Structure**: Historical performance standards, target metrics, acceptable ranges
**Scoring Applications**: Baseline scoring, target achievement assessment, performance validation

```bash
# Access performance baselines for scoring context
BASELINE_DATA=$(access_mcp_resource safla safla://performance-baselines)

# Extract baseline metrics
TARGET_RESPONSE_TIME=$(echo $BASELINE_DATA | jq '.target_response_time_ms')
TARGET_THROUGHPUT=$(echo $BASELINE_DATA | jq '.target_throughput_rps')
TARGET_CPU_USAGE=$(echo $BASELINE_DATA | jq '.target_cpu_utilization_percent')
TARGET_MEMORY_USAGE=$(echo $BASELINE_DATA | jq '.target_memory_utilization_percent')

# Score against baselines
RESPONSE_TIME_SCORE=$(echo "scale=2; if ($RESPONSE_TIME <= $TARGET_RESPONSE_TIME) 100 else 100 - (($RESPONSE_TIME - $TARGET_RESPONSE_TIME) / $TARGET_RESPONSE_TIME * 50)" | bc)
THROUGHPUT_SCORE=$(echo "scale=2; if ($THROUGHPUT >= $TARGET_THROUGHPUT) 100 else ($THROUGHPUT / $TARGET_THROUGHPUT * 100)" | bc)

echo "Response Time Score: $RESPONSE_TIME_SCORE/100"
echo "Throughput Score: $THROUGHPUT_SCORE/100"
```

### 2. Goal and Strategy Resources

#### safla://goals
**Purpose**: Current goal status and achievement data for goal scoring
**Data Structure**: Goal definitions, progress metrics, completion status, priority weights
**Scoring Applications**: Goal achievement scoring, portfolio analysis, progress tracking

```bash
# Access goals data for achievement scoring
GOALS_DATA=$(access_mcp_resource safla safla://goals)

# Calculate portfolio goal achievement score
TOTAL_WEIGHTED_SCORE=0
TOTAL_WEIGHT=0

for goal in $(echo $GOALS_DATA | jq -r '.goals[] | @base64'); do
    GOAL_INFO=$(echo $goal | base64 --decode)
    PROGRESS=$(echo $GOAL_INFO | jq '.progress_percentage')
    WEIGHT=$(echo $GOAL_INFO | jq '.priority_weight')
    QUALITY_SCORE=$(echo $GOAL_INFO | jq '.quality_score // 100')
    
    # Calculate weighted achievement score
    WEIGHTED_SCORE=$(echo "scale=2; $PROGRESS * $QUALITY_SCORE / 100 * $WEIGHT" | bc)
    TOTAL_WEIGHTED_SCORE=$(echo "scale=2; $TOTAL_WEIGHTED_SCORE + $WEIGHTED_SCORE" | bc)
    TOTAL_WEIGHT=$(echo "scale=2; $TOTAL_WEIGHT + $WEIGHT" | bc)
done

PORTFOLIO_SCORE=$(echo "scale=2; $TOTAL_WEIGHTED_SCORE / $TOTAL_WEIGHT" | bc)
echo "Portfolio Goal Achievement Score: $PORTFOLIO_SCORE/100"
```

#### safla://strategies
**Purpose**: Strategy performance data and effectiveness metrics
**Data Structure**: Strategy definitions, performance indicators, success rates, ROI metrics
**Scoring Applications**: Strategy effectiveness scoring, comparative analysis, optimization recommendations

```bash
# Access strategy data for effectiveness scoring
STRATEGY_DATA=$(access_mcp_resource safla safla://strategies)

# Score strategy effectiveness
for strategy in $(echo $STRATEGY_DATA | jq -r '.strategies[] | @base64'); do
    STRATEGY_INFO=$(echo $strategy | base64 --decode)
    STRATEGY_ID=$(echo $STRATEGY_INFO | jq -r '.id')
    SUCCESS_RATE=$(echo $STRATEGY_INFO | jq '.success_rate_percent')
    ROI=$(echo $STRATEGY_INFO | jq '.roi_percentage')
    IMPLEMENTATION_SCORE=$(echo $STRATEGY_INFO | jq '.implementation_quality_score')
    
    # Calculate composite strategy effectiveness score
    EFFECTIVENESS_SCORE=$(echo "scale=2; ($SUCCESS_RATE * 0.4) + (($ROI + 100) / 2 * 0.3) + ($IMPLEMENTATION_SCORE * 0.3)" | bc)
    
    echo "Strategy $STRATEGY_ID Effectiveness: $EFFECTIVENESS_SCORE/100"
done
```

#### safla://learning-metrics
**Purpose**: Learning progress and adaptation effectiveness data
**Data Structure**: Learning rates, accuracy metrics, retention scores, adaptation patterns
**Scoring Applications**: Learning effectiveness scoring, adaptation assessment, cognitive development tracking

```bash
# Access learning metrics for cognitive scoring
LEARNING_DATA=$(access_mcp_resource safla safla://learning-metrics)

# Extract learning performance metrics
ACCURACY_SCORE=$(echo $LEARNING_DATA | jq '.accuracy_percentage')
RETENTION_SCORE=$(echo $LEARNING_DATA | jq '.retention_percentage')
ADAPTATION_RATE=$(echo $LEARNING_DATA | jq '.adaptation_rate_score')
LEARNING_VELOCITY=$(echo $LEARNING_DATA | jq '.learning_velocity_score')

# Calculate comprehensive learning effectiveness score
LEARNING_EFFECTIVENESS=$(echo "scale=2; ($ACCURACY_SCORE * 0.3) + ($RETENTION_SCORE * 0.25) + ($ADAPTATION_RATE * 0.25) + ($LEARNING_VELOCITY * 0.2)" | bc)

echo "Learning Effectiveness Score: $LEARNING_EFFECTIVENESS/100"

# Analyze learning trends
TREND_DIRECTION=$(echo $LEARNING_DATA | jq -r '.trend_direction')
TREND_STRENGTH=$(echo $LEARNING_DATA | jq '.trend_strength_score')

echo "Learning Trend: $TREND_DIRECTION (Strength: $TREND_STRENGTH/100)"
```

### 3. Quality and Testing Resources

#### safla://test-results
**Purpose**: Test execution results and quality metrics
**Data Structure**: Test pass rates, coverage data, performance test results, quality indicators
**Scoring Applications**: Quality scoring, test effectiveness assessment, reliability metrics

```bash
# Access test results for quality scoring
TEST_DATA=$(access_mcp_resource safla safla://test-results)

# Extract test quality metrics
PASS_RATE=$(echo $TEST_DATA | jq '.overall_pass_rate_percent')
UNIT_TEST_COVERAGE=$(echo $TEST_DATA | jq '.unit_test_coverage_percent')
INTEGRATION_TEST_COVERAGE=$(echo $TEST_DATA | jq '.integration_test_coverage_percent')
PERFORMANCE_TEST_SCORE=$(echo $TEST_DATA | jq '.performance_test_score')

# Calculate comprehensive test quality score
TEST_QUALITY_SCORE=$(echo "scale=2; ($PASS_RATE * 0.4) + ($UNIT_TEST_COVERAGE * 0.25) + ($INTEGRATION_TEST_COVERAGE * 0.2) + ($PERFORMANCE_TEST_SCORE * 0.15)" | bc)

echo "Test Quality Score: $TEST_QUALITY_SCORE/100"

# Analyze test trend data
TEST_TREND=$(echo $TEST_DATA | jq '.quality_trend_7d')
echo "7-Day Quality Trend: $TEST_TREND"
```

#### safla://test-coverage
**Purpose**: Detailed test coverage analysis and metrics
**Data Structure**: Line coverage, branch coverage, function coverage, coverage trends
**Scoring Applications**: Coverage scoring, quality assessment, testing completeness evaluation

```bash
# Access test coverage data for coverage scoring
COVERAGE_DATA=$(access_mcp_resource safla safla://test-coverage)

# Extract coverage metrics
LINE_COVERAGE=$(echo $COVERAGE_DATA | jq '.line_coverage_percent')
BRANCH_COVERAGE=$(echo $COVERAGE_DATA | jq '.branch_coverage_percent')
FUNCTION_COVERAGE=$(echo $COVERAGE_DATA | jq '.function_coverage_percent')
STATEMENT_COVERAGE=$(echo $COVERAGE_DATA | jq '.statement_coverage_percent')

# Calculate weighted coverage score
COVERAGE_SCORE=$(echo "scale=2; ($LINE_COVERAGE * 0.3) + ($BRANCH_COVERAGE * 0.3) + ($FUNCTION_COVERAGE * 0.2) + ($STATEMENT_COVERAGE * 0.2)" | bc)

echo "Overall Coverage Score: $COVERAGE_SCORE/100"

# Identify coverage gaps
UNCOVERED_LINES=$(echo $COVERAGE_DATA | jq '.uncovered_lines_count')
CRITICAL_UNCOVERED=$(echo $COVERAGE_DATA | jq '.critical_uncovered_functions')

echo "Coverage Gaps: $UNCOVERED_LINES lines, $CRITICAL_UNCOVERED critical functions"
```

### 4. System Status and Health Resources

#### safla://status
**Purpose**: Current system status and health indicators
**Data Structure**: System health metrics, operational status, alert conditions
**Scoring Applications**: System health scoring, operational effectiveness assessment, availability metrics

```bash
# Access system status for health scoring
STATUS_DATA=$(access_mcp_resource safla safla://status)

# Extract health indicators
SYSTEM_HEALTH=$(echo $STATUS_DATA | jq '.overall_health_score')
UPTIME_PERCENTAGE=$(echo $STATUS_DATA | jq '.uptime_percentage_30d')
ERROR_RATE=$(echo $STATUS_DATA | jq '.error_rate_percentage')
RESPONSE_RELIABILITY=$(echo $STATUS_DATA | jq '.response_reliability_score')

# Calculate system reliability score
RELIABILITY_SCORE=$(echo "scale=2; ($SYSTEM_HEALTH * 0.3) + ($UPTIME_PERCENTAGE * 0.3) + ((100 - $ERROR_RATE) * 0.2) + ($RESPONSE_RELIABILITY * 0.2)" | bc)

echo "System Reliability Score: $RELIABILITY_SCORE/100"

# Check for active alerts
ACTIVE_ALERTS=$(echo $STATUS_DATA | jq '.active_alerts_count')
CRITICAL_ALERTS=$(echo $STATUS_DATA | jq '.critical_alerts_count')

if [ $CRITICAL_ALERTS -gt 0 ]; then
    echo "WARNING: $CRITICAL_ALERTS critical alerts active"
    RELIABILITY_SCORE=$(echo "scale=2; $RELIABILITY_SCORE * 0.8" | bc)  # Penalty for critical alerts
fi
```

#### safla://optimization-recommendations
**Purpose**: AI-generated optimization recommendations and improvement suggestions
**Data Structure**: Optimization opportunities, impact assessments, implementation priorities
**Scoring Applications**: Optimization potential scoring, improvement opportunity assessment, recommendation prioritization

```bash
# Access optimization recommendations for improvement scoring
OPT_DATA=$(access_mcp_resource safla safla://optimization-recommendations)

# Extract optimization metrics
TOTAL_RECOMMENDATIONS=$(echo $OPT_DATA | jq '.recommendations | length')
HIGH_IMPACT_COUNT=$(echo $OPT_DATA | jq '[.recommendations[] | select(.impact_level == "high")] | length')
IMPLEMENTATION_READINESS=$(echo $OPT_DATA | jq '.average_implementation_readiness_score')
POTENTIAL_IMPROVEMENT=$(echo $OPT_DATA | jq '.total_potential_improvement_percent')

# Calculate optimization opportunity score
OPTIMIZATION_SCORE=$(echo "scale=2; ($HIGH_IMPACT_COUNT / $TOTAL_RECOMMENDATIONS * 100 * 0.4) + ($IMPLEMENTATION_READINESS * 0.3) + ($POTENTIAL_IMPROVEMENT * 0.3)" | bc)

echo "Optimization Opportunity Score: $OPTIMIZATION_SCORE/100"
echo "Potential Improvement: $POTENTIAL_IMPROVEMENT%"
```

### 5. Deployment and Configuration Resources

#### safla://deployments
**Purpose**: Deployment status and performance data
**Data Structure**: Deployment metrics, success rates, rollback statistics, environment health
**Scoring Applications**: Deployment effectiveness scoring, release quality assessment, operational stability metrics

```bash
# Access deployment data for deployment scoring
DEPLOYMENT_DATA=$(access_mcp_resource safla safla://deployments)

# Extract deployment metrics
DEPLOYMENT_SUCCESS_RATE=$(echo $DEPLOYMENT_DATA | jq '.success_rate_30d_percent')
ROLLBACK_RATE=$(echo $DEPLOYMENT_DATA | jq '.rollback_rate_30d_percent')
DEPLOYMENT_FREQUENCY=$(echo $DEPLOYMENT_DATA | jq '.deployments_per_week')
LEAD_TIME_HOURS=$(echo $DEPLOYMENT_DATA | jq '.average_lead_time_hours')

# Calculate deployment effectiveness score
DEPLOYMENT_SCORE=$(echo "scale=2; ($DEPLOYMENT_SUCCESS_RATE * 0.4) + ((100 - $ROLLBACK_RATE) * 0.3) + (($DEPLOYMENT_FREQUENCY / 10) * 100 * 0.2) + ((168 - $LEAD_TIME_HOURS) / 168 * 100 * 0.1)" | bc)

echo "Deployment Effectiveness Score: $DEPLOYMENT_SCORE/100"

# Analyze deployment trends
DEPLOYMENT_TREND=$(echo $DEPLOYMENT_DATA | jq -r '.trend_direction')
echo "Deployment Trend: $DEPLOYMENT_TREND"
```

#### safla://config
**Purpose**: Current system configuration and settings
**Data Structure**: Configuration parameters, optimization settings, feature flags
**Scoring Applications**: Configuration optimization scoring, settings validation, compliance assessment

```bash
# Access configuration data for configuration scoring
CONFIG_DATA=$(access_mcp_resource safla safla://config)

# Extract configuration metrics
OPTIMIZATION_LEVEL=$(echo $CONFIG_DATA | jq '.optimization_level_percent')
SECURITY_SETTINGS_SCORE=$(echo $CONFIG_DATA | jq '.security_configuration_score')
PERFORMANCE_SETTINGS_SCORE=$(echo $CONFIG_DATA | jq '.performance_configuration_score')
COMPLIANCE_SCORE=$(echo $CONFIG_DATA | jq '.compliance_score')

# Calculate configuration effectiveness score
CONFIG_SCORE=$(echo "scale=2; ($OPTIMIZATION_LEVEL * 0.25) + ($SECURITY_SETTINGS_SCORE * 0.3) + ($PERFORMANCE_SETTINGS_SCORE * 0.25) + ($COMPLIANCE_SCORE * 0.2)" | bc)

echo "Configuration Effectiveness Score: $CONFIG_SCORE/100"
```

## Advanced Resource Integration Patterns

### 1. Multi-Resource Composite Scoring

#### Comprehensive System Scoring
```bash
#!/bin/bash
# Comprehensive system scoring using multiple resources

echo "Performing comprehensive system scoring analysis..."

# Performance dimension scoring
PERF_METRICS=$(access_mcp_resource safla safla://performance-metrics)
BENCHMARK_DATA=$(access_mcp_resource safla safla://benchmark-results)
BASELINE_DATA=$(access_mcp_resource safla safla://performance-baselines)

PERFORMANCE_SCORE=$(echo $PERF_METRICS | jq '.composite_performance_score')
BENCHMARK_SCORE=$(echo $BENCHMARK_DATA | jq '.latest_benchmark.overall_score')
BASELINE_ACHIEVEMENT=$(echo $BASELINE_DATA | jq '.baseline_achievement_percent')

PERFORMANCE_DIMENSION=$(echo "scale=2; ($PERFORMANCE_SCORE * 0.4) + ($BENCHMARK_SCORE * 0.3) + ($BASELINE_ACHIEVEMENT * 0.3)" | bc)

# Quality dimension scoring
TEST_RESULTS=$(access_mcp_resource safla safla://test-results)
COVERAGE_DATA=$(access_mcp_resource safla safla://test-coverage)

QUALITY_SCORE=$(echo $TEST_RESULTS | jq '.overall_quality_score')
COVERAGE_SCORE=$(echo $COVERAGE_DATA | jq '.weighted_coverage_score')

QUALITY_DIMENSION=$(echo "scale=2; ($QUALITY_SCORE * 0.6) + ($COVERAGE_SCORE * 0.4)" | bc)

# Goal achievement dimension scoring
GOALS_DATA=$(access_mcp_resource safla safla://goals)
STRATEGY_DATA=$(access_mcp_resource safla safla://strategies)

GOAL_ACHIEVEMENT=$(echo $GOALS_DATA | jq '.portfolio_achievement_score')
STRATEGY_EFFECTIVENESS=$(echo $STRATEGY_DATA | jq '.average_effectiveness_score')

ACHIEVEMENT_DIMENSION=$(echo "scale=2; ($GOAL_ACHIEVEMENT * 0.7) + ($STRATEGY_EFFECTIVENESS * 0.3)" | bc)

# System health dimension scoring
STATUS_DATA=$(access_mcp_resource safla safla://status)
DEPLOYMENT_DATA=$(access_mcp_resource safla safla://deployments)

HEALTH_SCORE=$(echo $STATUS_DATA | jq '.overall_health_score')
DEPLOYMENT_SCORE=$(echo $DEPLOYMENT_DATA | jq '.deployment_effectiveness_score')

HEALTH_DIMENSION=$(echo "scale=2; ($HEALTH_SCORE * 0.6) + ($DEPLOYMENT_SCORE * 0.4)" | bc)

# Calculate overall system score
OVERALL_SYSTEM_SCORE=$(echo "scale=2; ($PERFORMANCE_DIMENSION * 0.3) + ($QUALITY_DIMENSION * 0.25) + ($ACHIEVEMENT_DIMENSION * 0.25) + ($HEALTH_DIMENSION * 0.2)" | bc)

echo "System Scoring Results:"
echo "  Performance Dimension: $PERFORMANCE_DIMENSION/100"
echo "  Quality Dimension: $QUALITY_DIMENSION/100"
echo "  Achievement Dimension: $ACHIEVEMENT_DIMENSION/100"
echo "  Health Dimension: $HEALTH_DIMENSION/100"
echo "  Overall System Score: $OVERALL_SYSTEM_SCORE/100"
```

### 2. Trend Analysis and Predictive Scoring

#### Historical Trend Integration
```bash
#!/bin/bash
# Historical trend analysis for predictive scoring

echo "Analyzing historical trends for predictive scoring..."

# Collect historical performance data
PERF_HISTORY=$(access_mcp_resource safla safla://performance-metrics | jq '.historical_data_30d')
QUALITY_HISTORY=$(access_mcp_resource safla safla://test-results | jq '.historical_quality_30d')
GOAL_HISTORY=$(access_mcp_resource safla safla://goals | jq '.historical_achievement_30d')

# Calculate trend coefficients
PERF_TREND=$(echo $PERF_HISTORY | jq '[.[] | .score] | (.[length-1] - .[0]) / length')
QUALITY_TREND=$(echo $QUALITY_HISTORY | jq '[.[] | .score] | (.[length-1] - .[0]) / length')
GOAL_TREND=$(echo $GOAL_HISTORY | jq '[.[] | .score] | (.[length-1] - .[0]) / length')

# Project future scores based on trends
CURRENT_PERF=$(access_mcp_resource safla safla://performance-metrics | jq '.current_score')
CURRENT_QUALITY=$(access_mcp_resource safla safla://test-results | jq '.current_score')
CURRENT_GOALS=$(access_mcp_resource safla safla://goals | jq '.current_score')

PROJECTED_PERF_30D=$(echo "scale=2; $CURRENT_PERF + ($PERF_TREND * 30)" | bc)
PROJECTED_QUALITY_30D=$(echo "scale=2; $CURRENT_QUALITY + ($QUALITY_TREND * 30)" | bc)
PROJECTED_GOALS_30D=$(echo "scale=2; $CURRENT_GOALS + ($GOAL_TREND * 30)" | bc)

echo "Trend Analysis Results:"
echo "  Performance Trend: $PERF_TREND points/day"
echo "  Quality Trend: $QUALITY_TREND points/day"
echo "  Goals Trend: $GOAL_TREND points/day"
echo ""
echo "30-Day Projections:"
echo "  Projected Performance: $PROJECTED_PERF_30D/100"
echo "  Projected Quality: $PROJECTED_QUALITY_30D/100"
echo "  Projected Goals: $PROJECTED_GOALS_30D/100"
```

### 3. Real-Time Resource Monitoring

#### Continuous Resource-Based Scoring
```bash
#!/bin/bash
# Continuous resource monitoring for real-time scoring

echo "Starting continuous resource-based scoring monitor..."

MONITORING_INTERVAL=60  # 1 minute
LOG_FILE="resource_scores.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Collect current scores from multiple resources
    PERF_SCORE=$(access_mcp_resource safla safla://performance-metrics | jq '.current_score')
    HEALTH_SCORE=$(access_mcp_resource safla safla://status | jq '.overall_health_score')
    QUALITY_SCORE=$(access_mcp_resource safla safla://test-results | jq '.latest_quality_score')
    
    # Calculate real-time composite score
    COMPOSITE_SCORE=$(echo "scale=2; ($PERF_SCORE * 0.4) + ($HEALTH_SCORE * 0.3) + ($QUALITY_SCORE * 0.3)" | bc)
    
    # Log scores
    echo "$TIMESTAMP,$PERF_SCORE,$HEALTH_SCORE,$QUALITY_SCORE,$COMPOSITE_SCORE" >> $LOG_FILE
    
    echo "[$TIMESTAMP] Composite Score: $COMPOSITE_SCORE/100 (P:$PERF_SCORE H:$HEALTH_SCORE Q:$QUALITY_SCORE)"
    
    # Check for significant score changes
    if [ -f "$LOG_FILE" ]; then
        PREVIOUS_SCORE=$(tail -2 $LOG_FILE | head -1 | cut -d',' -f5)
        SCORE_CHANGE=$(echo "scale=2; $COMPOSITE_SCORE - $PREVIOUS_SCORE" | bc)
        
        if (( $(echo "$SCORE_CHANGE > 5 || $SCORE_CHANGE < -5" | bc -l) )); then
            echo "ALERT: Significant score change detected: $SCORE_CHANGE points"
        fi
    fi
    
    sleep $MONITORING_INTERVAL
done
```

## Resource Data Validation and Quality Assurance

### 1. Data Integrity Validation

#### Resource Data Validation
```bash
#!/bin/bash
# Validate resource data integrity for scoring accuracy

validate_resource_data() {
    local resource_uri=$1
    local expected_fields=$2
    
    echo "Validating resource: $resource_uri"
    
    # Access resource data
    if RESOURCE_DATA=$(access_mcp_resource safla "$resource_uri" 2>/dev/null); then
        # Check for required fields
        for field in $expected_fields; do
            if ! echo $RESOURCE_DATA | jq -e ".$field" >/dev/null 2>&1; then
                echo "ERROR: Missing required field '$field' in $resource_uri"
                return 1
            fi
        done
        
        # Validate data types and ranges
        if echo $RESOURCE_DATA | jq -e '.score' >/dev/null 2>&1; then
            SCORE=$(echo $RESOURCE_DATA | jq '.score')
            if (( $(echo "$SCORE < 0 || $SCORE > 100" | bc -l) )); then
                echo "WARNING: Score out of range (0-100): $SCORE"
            fi
        fi
        
        echo "Resource $resource_uri validation passed"
        return 0
    else
        echo "ERROR: Unable to access resource $resource_uri"
        return 1
    fi
}

# Validate critical resources
validate_resource_data "safla://performance-metrics" "current_score cpu_utilization_percent memory_usage_percent"
validate_resource_data "safla://test-results" "overall_quality_score pass_rate_percent"
validate_resource_data "safla://goals" "portfolio_achievement_score"
validate_resource_data "safla://status" "overall_health_score uptime_percentage_30d"
```

### 2. Resource Availability Monitoring

#### Resource Health Monitoring
```bash
#!/bin/bash
# Monitor resource availability and health

monitor_resource_health() {
    local resources=("safla://performance-metrics" "safla://test-results" "safla://goals" "safla://status" "safla://strategies")
    local health_log="resource_health.log"
    
    echo "Monitoring resource health..."
    
    for resource in "${resources[@]}"; do
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        
        if access_mcp_resource safla "$resource" >/dev/null 2>&1; then
            STATUS="HEALTHY"
            RESPONSE_TIME=$(time (access_mcp_resource safla "$resource" >/dev/null 2>&1) 2>&1 | grep real | awk '{print $2}')
        else
            STATUS="UNHEALTHY"
            RESPONSE_TIME="N/A"
        fi
        
        echo "$TIMESTAMP,$resource,$STATUS,$RESPONSE_TIME" >> $health_log
        echo "[$TIMESTAMP] $resource: $STATUS ($RESPONSE_TIME)"
    done
}

# Run resource health monitoring
monitor_resource_health
```

This comprehensive resources documentation provides the Scorer mode with detailed guidance for leveraging SAFLA's resource system to implement effective data-driven scoring and evaluation workflows within the aiGI framework.