#!/usr/bin/env nu
# Performance Analytics System for Polyglot Development Environment
# Provides intelligent monitoring of build times, resource usage, and performance metrics

# Initialize performance metrics database
def "perf init" [] {
    if not (".performance" | path exists) {
        mkdir .performance
    }
    
    if not (".performance/metrics.jsonl" | path exists) {
        [] | save .performance/metrics.jsonl
    }
    
    if not (".performance/config.json" | path exists) {
        {
            version: "1.0",
            environments: ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env"],
            retention_days: 30,
            alert_thresholds: {
                build_time_increase: 0.5,  # 50% increase triggers alert
                memory_usage: 0.8,         # 80% memory usage
                test_duration_increase: 0.3 # 30% test duration increase
            }
        } | save .performance/config.json
    }
}

# Record performance metric
def "perf record" [
    event_type: string,    # build, test, lint, format, etc.
    environment: string,   # python-env, typescript-env, etc.
    duration: duration,    # execution time
    --status: string = "success",  # success, failure, timeout
    --details: record = {}         # additional context
] {
    perf init
    
    let metric = {
        timestamp: (date now),
        event_type: $event_type,
        environment: $environment,
        duration_ms: ($duration / 1ms),  # Convert to milliseconds
        status: $status,
        details: $details,
        resource_usage: (sys | select host.memory host.cpu)
    }
    
    $metric | to json --raw | save --append .performance/metrics.jsonl
    
    # Check for performance alerts
    perf check-alerts $event_type $environment $duration
}

# Measure and record command execution
def "perf measure" [
    event_type: string,
    environment: string,
    command: string,
    --quiet = false
] {
    if not $quiet {
        print $"🔍 Measuring ($event_type) in ($environment)..."
    }
    
    let start_time = (date now)
    
    # Execute command and capture result
    let result = (try {
        nu -c $command | complete
    } catch {
        {stdout: "", stderr: $in, exit_code: 1}
    })
    
    let end_time = (date now)
    let duration = $end_time - $start_time
    
    let status = if $result.exit_code == 0 { "success" } else { "failure" }
    let details = {
        command: $command,
        stdout_lines: ($result.stdout | lines | length),
        stderr_lines: ($result.stderr | lines | length),
        exit_code: $result.exit_code
    }
    
    perf record $event_type $environment $duration --status $status --details $details
    
    if not $quiet {
        let duration_str = if $duration > 1sec { 
            $"($duration / 1sec | math round)s" 
        } else { 
            $"($duration / 1ms | math round)ms" 
        }
        
        let status_emoji = if $status == "success" { "✅" } else { "❌" }
        print $"($status_emoji) ($event_type) completed in ($duration_str)"
    }
    
    $result
}

# Check for performance alerts
def "perf check-alerts" [event_type: string, environment: string, duration: duration] {
    let config = open .performance/config.json
    let threshold = $config.alert_thresholds.build_time_increase
    
    # Get recent metrics for comparison
    let recent_metrics = (
        open .performance/metrics.jsonl 
        | lines 
        | each { |line| $line | from json }
        | where event_type == $event_type and environment == $environment
        | where timestamp > ((date now) - 7day)
        | sort-by timestamp
    )
    
    if ($recent_metrics | length) > 5 {
        let avg_duration = ($recent_metrics | get duration_ms | math avg)
        let current_duration = ($duration | into int) / 1000000
        
        if $current_duration > ($avg_duration * (1 + $threshold)) {
            let increase_pct = (($current_duration - $avg_duration) / $avg_duration * 100 | math round)
            print $"⚠️  Performance Alert: ($event_type) in ($environment) took ($increase_pct)% longer than average"
            
            # Log alert
            {
                type: "performance_degradation",
                event_type: $event_type,
                environment: $environment,
                current_duration_ms: $current_duration,
                average_duration_ms: $avg_duration,
                increase_percentage: $increase_pct,
                timestamp: (date now)
            } | to json --raw | save --append .performance/alerts.jsonl
        }
    }
}

# Generate performance report
def "perf report" [
    --days: int = 7,        # Number of days to analyze
    --environment: string = "all",  # Specific environment or "all"
    --format: string = "table"      # Output format: table, json, chart
] {
    perf init
    
    let start_date = (date now) - ($days * 1day)
    
    let metrics = (
        open .performance/metrics.jsonl 
        | lines 
        | each { |line| $line | from json }
        | where timestamp > $start_date
    )
    
    let filtered_metrics = if $environment == "all" {
        $metrics
    } else {
        $metrics | where environment == $environment
    }
    
    match $format {
        "table" => {
            print $"📊 Performance Report - Last ($days) days"
            print "=" * 50
            
            # Summary by environment
            let summary = (
                $filtered_metrics 
                | group-by environment 
                | transpose environment data
                | each { |row|
                    let env_data = $row.data
                    {
                        environment: $row.environment,
                        total_events: ($env_data | length),
                        avg_duration_ms: ($env_data | get duration_ms | math avg | math round),
                        success_rate: (($env_data | where status == "success" | length) / ($env_data | length) * 100 | math round),
                        slowest_event: ($env_data | sort-by duration_ms | last | get event_type),
                        slowest_duration_ms: ($env_data | sort-by duration_ms | last | get duration_ms)
                    }
                }
            )
            
            $summary | table
            
            # Performance trends
            print "\n🔄 Performance Trends by Event Type:"
            let trends = (
                $filtered_metrics
                | group-by event_type
                | transpose event_type data
                | each { |row|
                    let event_data = $row.data | sort-by timestamp
                    let first_half = ($event_data | first (($event_data | length) / 2 | math floor))
                    let second_half = ($event_data | last (($event_data | length) / 2 | math ceil))
                    
                    let first_avg = if ($first_half | length) > 0 { $first_half | get duration_ms | math avg } else { 0 }
                    let second_avg = if ($second_half | length) > 0 { $second_half | get duration_ms | math avg } else { 0 }
                    
                    let trend = if $first_avg > 0 {
                        (($second_avg - $first_avg) / $first_avg * 100 | math round)
                    } else { 0 }
                    
                    {
                        event_type: $row.event_type,
                        count: ($event_data | length),
                        avg_duration_ms: ($event_data | get duration_ms | math avg | math round),
                        trend_percentage: $trend,
                        trend_direction: (if $trend > 5 { "🔴 Slower" } else if $trend < -5 { "🟢 Faster" } else { "➡️ Stable" })
                    }
                }
            )
            
            $trends | table
        },
        "json" => {
            $filtered_metrics | to json
        },
        _ => {
            print "Invalid format. Use: table, json"
        }
    }
}

# Clean old performance data
def "perf cleanup" [--days: int = 30] {
    perf init
    
    let cutoff_date = (date now) - ($days * 1day)
    
    let current_metrics = (
        open .performance/metrics.jsonl 
        | lines 
        | each { |line| $line | from json }
        | where timestamp > $cutoff_date
    )
    
    $current_metrics | each { |metric| $metric | to json --raw } | str join "\n" | save .performance/metrics.jsonl
    
    print $"🧹 Cleaned performance data older than ($days) days"
}

# Performance dashboard - real-time monitoring
def "perf dashboard" [] {
    while true {
        clear
        print "🎯 Real-Time Performance Dashboard"
        print "================================"
        print $"Last updated: (date now | format date '%Y-%m-%d %H:%M:%S')"
        print ""
        
        perf report --days 1 --format table
        
        print "\n📈 Resource Usage:"
        let memory = (sys | get host.memory)
        let memory_pct = ($memory.used / $memory.total * 100 | math round)
        print $"Memory: ($memory_pct)% used (($memory.used | into string) / ($memory.total | into string))"
        
        print "\nPress Ctrl+C to exit dashboard"
        sleep 10sec
    }
}

# Performance optimization suggestions
def "perf optimize" [environment?: string] {
    perf init
    
    let metrics = (
        open .performance/metrics.jsonl 
        | lines 
        | each { |line| $line | from json }
        | where timestamp > ((date now) - 7day)
    )
    
    let env_metrics = if $environment != null {
        $metrics | where environment == $environment
    } else {
        $metrics
    }
    
    print "🚀 Performance Optimization Suggestions"
    print "======================================"
    
    # Identify slowest operations
    let slowest = (
        $env_metrics 
        | sort-by duration_ms 
        | last 5
        | each { |metric|
            $"• ($metric.event_type) in ($metric.environment): ($metric.duration_ms)ms - Consider optimizing"
        }
    )
    
    if ($slowest | length) > 0 {
        print "\n📌 Slowest Operations (Last 7 days):"
        $slowest | each { |suggestion| print $suggestion }
    }
    
    # Check for frequent failures
    let failures = (
        $env_metrics 
        | where status == "failure"
        | group-by event_type
        | transpose event_type failures
        | where ($it.failures | length) > 3
        | each { |row|
            $"• ($row.event_type): ($row.failures | length) failures - Review error patterns"
        }
    )
    
    if ($failures | length) > 0 {
        print "\n⚠️  High Failure Rate Operations:"
        $failures | each { |suggestion| print $suggestion }
    }
    
    # Memory usage recommendations
    let high_memory = (
        $env_metrics 
        | where ($in.resource_usage.host.memory.used / $in.resource_usage.host.memory.total) > 0.8
        | group-by environment
        | transpose environment count
        | each { |row|
            $"• ($row.environment): High memory usage detected ($row.count | length) times"
        }
    )
    
    if ($high_memory | length) > 0 {
        print "\n💾 Memory Usage Recommendations:"
        $high_memory | each { |suggestion| print $suggestion }
    }
}

# Main command dispatcher
def main [command: string, ...args] {
    match $command {
        "init" => { perf init },
        "record" => { 
            if ($args | length) >= 3 {
                perf record $args.0 $args.1 ($args.2 | into duration)
            } else {
                print "Usage: perf record <event_type> <environment> <duration>"
            }
        },
        "measure" => {
            if ($args | length) >= 3 {
                perf measure $args.0 $args.1 $args.2
            } else {
                print "Usage: perf measure <event_type> <environment> <command>"
            }
        },
        "report" => { perf report },
        "cleanup" => { perf cleanup },
        "dashboard" => { perf dashboard },
        "optimize" => { perf optimize },
        _ => {
            print "Performance Analytics System"
            print "Usage:"
            print "  perf init                     - Initialize performance tracking"
            print "  perf record <type> <env> <dur> - Record a performance metric"
            print "  perf measure <type> <env> <cmd> - Measure and record command execution"
            print "  perf report [--days N]        - Generate performance report"
            print "  perf cleanup [--days N]       - Clean old performance data"
            print "  perf dashboard               - Real-time performance dashboard"
            print "  perf optimize [environment]  - Get optimization suggestions"
        }
    }
}