# File: scripts/uap-tools.nu
# UAP Development Utilities in Nushell

# Get agent status across all frameworks
export def "uap agent-status" [] {
    let status = (
        http get "http://localhost:8000/api/status" 
        | from json
    )
    
    $status | table
}

# Monitor agent performance in real-time
export def "uap monitor" [
    --agent_id: string = "all"  # Specific agent ID or "all"
    --duration: string = "60"   # Monitoring duration in seconds
    --interval: string = "5"    # Update interval in seconds
] {
    let duration_sec = ($duration | into int)
    let interval_sec = ($interval | into int)
    let end_time = (date now) | date to-record | get epoch
    let target_end = ($end_time + $duration_sec)
    
    print $"🔍 Monitoring UAP system (Duration: ($duration)s, Interval: ($interval_sec)s)"
    print "Press Ctrl+C to stop monitoring\n"
    
    while (date now | date to-record | get epoch) < $target_end {
        clear
        print "=== UAP System Monitor ==="
        print $"Time: (date now | format date '%Y-%m-%d %H:%M:%S')\n"
        
        # Get system health
        try {
            let health = (http get "http://localhost:8000/api/monitoring/health" | from json)
            
            print "🏥 System Health:"
            print $"  Overall: (if $health.overall_healthy { '✅ Healthy' } else { '❌ Unhealthy' })"
            print $"  CPU: (if $health.system_health.cpu_healthy { '✅' } else { '⚠️' }) ($health.current_stats.cpu_percent)%"
            print $"  Memory: (if $health.system_health.memory_healthy { '✅' } else { '⚠️' }) ($health.current_stats.memory_percent)%"
            print $"  Connections: ($health.current_stats.active_connections)"
            print ""
            
            # Get agent performance
            let agents = (http get "http://localhost:8000/api/monitoring/agents" | from json)
            
            if ($agents | length) > 0 {
                print "🤖 Agent Performance:"
                for agent in ($agents | transpose key value) {
                    let stats = $agent.value
                    print $"  ($stats.framework)/($stats.agent_id):"
                    print $"    Requests: ($stats.total_requests)"
                    print $"    Avg Response: ($stats.avg_response_time_ms | math round)ms"
                    print $"    Success Rate: ($stats.success_rate | math round)%"
                }
                print ""
            }
            
            # Get WebSocket stats
            let ws_stats = (http get "http://localhost:8000/api/monitoring/websockets" | from json)
            print $"🔌 WebSocket Connections: ($ws_stats.total_active_connections)"
            
        } catch {
            print "❌ Error fetching monitoring data - is the UAP backend running?"
        }
        
        sleep $"($interval_sec)sec"
    }
    
    print "\n✅ Monitoring session complete."
}

# Generate comprehensive performance report
export def "uap generate-report" [
    --format: string = "json"    # Output format (json, table, csv)
    --time_window: string = "60" # Time window in minutes
    --output: string = ""        # Output file path (optional)
] {
    print $"📊 Generating UAP performance report (Time window: ($time_window)m, Format: ($format))"
    
    try {
        # Get comprehensive performance data
        let overview = (http get "http://localhost:8000/api/monitoring/overview" | from json)
        let performance = (http get $"http://localhost:8000/api/monitoring/performance?time_window=($time_window)" | from json)
        let agents = (http get "http://localhost:8000/api/monitoring/agents" | from json)
        let websockets = (http get "http://localhost:8000/api/monitoring/websockets" | from json)
        
        let report = {
            generated_at: (date now | format date '%Y-%m-%d %H:%M:%S'),
            time_window_minutes: ($time_window | into int),
            system_overview: {
                overall_healthy: $overview.system_health.overall_healthy,
                active_agents: $overview.active_agents,
                active_connections: $overview.active_connections,
                avg_response_time_ms: ($overview.avg_response_time_ms | math round),
                error_rate_percent: ($overview.error_rate_percent | math round)
            },
            agent_performance: $agents,
            websocket_stats: {
                total_connections: $websockets.total_active_connections,
                connections_by_agent: $websockets.connections_by_agent
            },
            system_health: $overview.system_health.current_stats,
            metrics_summary: $performance.metrics_summary
        }
        
        # Output in requested format
        let formatted_output = match $format {
            "json" => ($report | to json),
            "table" => {
                print "=== UAP Performance Report ==="
                print $"Generated: ($report.generated_at)"
                print $"Time Window: ($report.time_window_minutes) minutes\n"
                
                print "📈 System Overview:"
                $report.system_overview | table
                print ""
                
                print "🤖 Agent Performance:"
                if ($report.agent_performance | length) > 0 {
                    $report.agent_performance | transpose key value | table
                } else {
                    print "  No agent data available"
                }
                print ""
                
                print "🔌 WebSocket Stats:"
                $report.websocket_stats | table
                
                $report | to json  # Also return JSON for potential file output
            },
            "csv" => {
                # Convert agent performance to CSV format
                if ($report.agent_performance | length) > 0 {
                    $report.agent_performance | transpose key value | to csv
                } else {
                    "agent_id,framework,total_requests,avg_response_time_ms,success_rate\n"
                }
            },
            _ => {
                print $"❌ Unsupported format: ($format). Supported: json, table, csv"
                return
            }
        }
        
        # Save to file if output path specified
        if ($output | str length) > 0 {
            $formatted_output | save $output
            print $"✅ Report saved to: ($output)"
        } else {
            $formatted_output
        }
        
    } catch {
        print "❌ Error generating report - is the UAP backend running?"
        return null
    }
}

# Deploy to SkyPilot
export def "uap deploy" [
    --env: string = "production"  # Environment (production, staging, dev)
] {
    print $"Deploying to ($env) environment..."
    cd ..
    skypilot up -c $"skypilot/uap-($env).yaml"
}

# Check comprehensive system health
export def "uap health-check" [] {
    try {
        print "🏥 UAP System Health Check"
        print "========================\n"
        
        # Basic health endpoint
        let basic_health = (http get "http://localhost:8000/health" | from json)
        print $"✅ Basic Health: ($basic_health.status)"
        print $"   Timestamp: ($basic_health.timestamp)\n"
        
        # Detailed monitoring health
        let detailed_health = (http get "http://localhost:8000/api/monitoring/health" | from json)
        
        print "🖥️ System Resources:"
        print $"   Overall Health: (if $detailed_health.overall_healthy { '✅ Healthy' } else { '❌ Unhealthy' })"
        print $"   CPU Usage: ($detailed_health.current_stats.cpu_percent)% (Threshold: ($detailed_health.thresholds.cpu_usage_percent)%)"
        print $"   Memory Usage: ($detailed_health.current_stats.memory_percent)% (Threshold: ($detailed_health.thresholds.memory_usage_percent)%)"
        print $"   Active Connections: ($detailed_health.current_stats.active_connections)"
        print ""
        
        print "🤖 Agent Health:"
        for agent in ($detailed_health.agent_health | transpose key value) {
            let agent_data = $agent.value
            let status_icon = if $agent_data.healthy { "✅" } else { "⚠️" }
            print $"   ($agent.key): ($status_icon)"
            print $"     Response Time P95: ($agent_data.response_time_p95 | math round)ms"
            print $"     Success Rate: ($agent_data.success_rate | math round)%"
        }
        print ""
        
        print "📊 Performance Thresholds:"
        print $"   Agent Response Time (P95): <($detailed_health.thresholds.agent_response_time_p95_ms)ms"
        print $"   Max Concurrent Sessions: ($detailed_health.thresholds.max_concurrent_sessions)"
        print $"   Error Rate: <($detailed_health.thresholds.error_rate_percent)%"
        
    } catch {
        print "❌ Health check failed - is the UAP backend running on port 8000?"
    }
}

# Get real-time metrics for a specific metric
export def "uap metrics" [
    metric_name: string          # Metric name to query
    --time_window: string = "60" # Time window in minutes
    --tags: string = ""          # Filter tags in JSON format
] {
    try {
        print $"📈 Fetching metric: ($metric_name)"
        
        let url = if ($tags | str length) > 0 {
            $"http://localhost:8000/api/monitoring/metrics/($metric_name)?time_window=($time_window)&tags=($tags)"
        } else {
            $"http://localhost:8000/api/monitoring/metrics/($metric_name)?time_window=($time_window)"
        }
        
        let metric_data = (http get $url | from json)
        
        print $"Metric: ($metric_data.metric_name)"
        print $"Unit: ($metric_data.unit)"
        print $"Time Range: ($metric_data.time_range)"
        print $"Data Points: (($metric_data.data_points | length))\n"
        
        if ($metric_data.data_points | length) > 0 {
            # Calculate statistics
            let values = ($metric_data.data_points | get value)
            let avg = ($values | math avg)
            let min_val = ($values | math min)
            let max_val = ($values | math max)
            
            print "📊 Statistics:"
            print $"   Average: ($avg | math round)"
            print $"   Minimum: ($min_val)"
            print $"   Maximum: ($max_val)"
            print ""
            
            # Show recent data points
            print "🕒 Recent Data Points:"
            $metric_data.data_points | last 10 | select timestamp value | table
        } else {
            print "No data points found for this metric in the specified time window."
        }
        
    } catch {
        print $"❌ Error fetching metric ($metric_name) - check if the metric exists and UAP backend is running"
    }
}

# List all available metrics
export def "uap metrics-list" [] {
    try {
        let metrics = (http get "http://localhost:8000/api/monitoring/metrics" | from json)
        
        print "📊 Available Metrics:"
        print "=====================\n"
        
        for metric in $metrics {
            print $"• ($metric)"
        }
        
        print $"\nTotal metrics available: (($metrics | length))"
        print "\nUse 'uap metrics <metric_name>' to view specific metric data."
        
    } catch {
        print "❌ Error fetching metrics list - is the UAP backend running?"
    }
}

# View Prometheus metrics
export def "uap prometheus" [] {
    try {
        print "📊 Prometheus Metrics Endpoint"
        print "==============================\n"
        
        let prometheus_data = (http get "http://localhost:8000/api/monitoring/prometheus")
        
        # Count metrics
        let lines = ($prometheus_data | split row "\n")
        let metric_lines = ($lines | where $it =~ "^[a-zA-Z]" and not ($it | str starts-with "#"))
        
        print $"📈 Total metric lines: (($metric_lines | length))"
        print $"📝 Total lines (including comments): (($lines | length))"
        print ""
        print "Sample metrics (first 10 lines):"
        print "================================"
        
        $lines | first 10 | each { |line| print $line }
        
        print "\n💡 Access full Prometheus metrics at: http://localhost:8000/api/monitoring/prometheus"
        
    } catch {
        print "❌ Error fetching Prometheus metrics - is the UAP backend running?"
    }
}