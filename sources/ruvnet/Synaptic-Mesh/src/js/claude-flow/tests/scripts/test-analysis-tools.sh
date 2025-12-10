#!/bin/bash
source test-mcp-tools-simple.sh
CATEGORY="analysis"

log_info "📊 Testing Analysis & Monitoring Tools (13 tools)"
test_tool "task_status" '{"taskId":"test-task"}'
test_tool "task_results" '{"taskId":"test-task"}'
test_tool "benchmark_run" '{"suite":"performance"}'
test_tool "bottleneck_analyze" '{"component":"test_component"}'
test_tool "performance_report" '{"timeframe":"24h","format":"summary"}'
test_tool "token_usage" '{"operation":"test_op"}'
test_tool "metrics_collect" '{"components":["cpu","memory"]}'
test_tool "trend_analysis" '{"metric":"response_time","period":"7d"}'
test_tool "cost_analysis" '{"timeframe":"30d"}'
test_tool "quality_assess" '{"target":"test_system","criteria":["performance","reliability"]}'
test_tool "error_analysis" '{"logs":["error1","error2"]}'
test_tool "usage_stats" '{"component":"api"}'
test_tool "health_check" '{"components":["database","api","cache"]}'
log_success "📊 Analysis category testing complete!"