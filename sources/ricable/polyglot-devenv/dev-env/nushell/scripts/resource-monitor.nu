#!/usr/bin/env nu
# Resource Usage Monitoring System for Polyglot Development Environment
# Monitors memory, CPU, disk usage and provides intelligent alerts

# Initialize resource monitoring
def "resource init" [] {
    mkdir .performance
    
    if not (".performance/resource_usage.jsonl" | path exists) {
        [] | save .performance/resource_usage.jsonl
    }
    
    if not (".performance/resource_config.json" | path exists) {
        {
            version: "1.0",
            monitoring_interval_seconds: 30,
            alert_thresholds: {
                memory_usage_percent: 85,
                cpu_usage_percent: 90,
                disk_usage_percent: 90,
                swap_usage_percent: 50
            },
            retention_hours: 72,
            environments: ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env"]
        } | save .performance/resource_config.json
    }
}

# Collect current resource usage snapshot
def "resource snapshot" [context?: string] {
    let system_info = (sys)
    let timestamp = (date now)
    
    let snapshot = {
        timestamp: $timestamp,
        context: ($context | default "general"),
        memory: {
            total_gb: ($system_info.host.memory.total / 1GB | math round --precision 2),
            used_gb: ($system_info.host.memory.used / 1GB | math round --precision 2),
            available_gb: ($system_info.host.memory.available / 1GB | math round --precision 2),
            usage_percent: ($system_info.host.memory.used / $system_info.host.memory.total * 100 | math round --precision 1),
            swap_total_gb: ($system_info.host.memory.swap_total / 1GB | math round --precision 2),
            swap_used_gb: ($system_info.host.memory.swap_used / 1GB | math round --precision 2),
            swap_usage_percent: (if $system_info.host.memory.swap_total > 0 { 
                $system_info.host.memory.swap_used / $system_info.host.memory.swap_total * 100 | math round --precision 1 
            } else { 0 })
        },
        cpu: {
            usage_percent: ($system_info.host.cpu | each { |core| $core.cpu_usage } | math avg | math round --precision 1),
            core_count: ($system_info.host.cpu | length),
            load_average: (if ($system_info.host | get load_avg? | default null) != null { $system_info.host.load_avg } else { null })
        },
        disk: {
            filesystems: ($system_info.host.disks | each { |disk|
                {
                    name: $disk.name,
                    mount: $disk.mount,
                    total_gb: ($disk.total / 1GB | math round --precision 2),
                    used_gb: (($disk.total - $disk.available) / 1GB | math round --precision 2),
                    available_gb: ($disk.available / 1GB | math round --precision 2),
                    usage_percent: (($disk.total - $disk.available) / $disk.total * 100 | math round --precision 1)
                }
            })
        },
        processes: {
            total_count: ($system_info.host.processes | length),
            high_memory_processes: ($system_info.host.processes 
                | where memory > 100MB 
                | sort-by memory 
                | last 5
                | each { |proc| 
                    {
                        name: $proc.name,
                        pid: $proc.pid,
                        memory_mb: ($proc.memory / 1MB | math round --precision 1),
                        cpu_percent: ($proc.cpu_usage | math round --precision 1)
                    }
                }
            )
        }
    }
    
    $snapshot
}

# Record resource usage with context
def "resource record" [context: string = "general"] {
    resource init
    
    let snapshot = (resource snapshot $context)
    $snapshot | to json --raw | save --append .performance/resource_usage.jsonl
    
    # Check for alerts
    resource check-alerts $snapshot
    
    $snapshot
}

# Check for resource usage alerts
def "resource check-alerts" [snapshot: record] {
    let config = (open .performance/resource_config.json)
    let alerts = []
    
    # Memory alerts
    if $snapshot.memory.usage_percent > $config.alert_thresholds.memory_usage_percent {
        let alert = {
            type: "high_memory_usage",
            severity: "warning",
            message: $"Memory usage at ($snapshot.memory.usage_percent)% (($snapshot.memory.used_gb)GB used)",
            threshold: $config.alert_thresholds.memory_usage_percent,
            current_value: $snapshot.memory.usage_percent,
            timestamp: $snapshot.timestamp,
            recommendations: [
                "Close unused applications",
                "Check for memory leaks in development processes",
                "Consider upgrading system memory"
            ]
        }
        $alerts | append $alert
        print $"⚠️  Memory Alert: ($alert.message)"
    }
    
    # Swap usage alerts
    if $snapshot.memory.swap_usage_percent > $config.alert_thresholds.swap_usage_percent {
        let alert = {
            type: "high_swap_usage",
            severity: "warning",
            message: $"Swap usage at ($snapshot.memory.swap_usage_percent)% (($snapshot.memory.swap_used_gb)GB used)",
            threshold: $config.alert_thresholds.swap_usage_percent,
            current_value: $snapshot.memory.swap_usage_percent,
            timestamp: $snapshot.timestamp,
            recommendations: [
                "Free up physical memory",
                "Restart memory-intensive applications",
                "Add more RAM to reduce swap dependency"
            ]
        }
        $alerts | append $alert
        print $"🔶 Swap Alert: ($alert.message)"
    }
    
    # CPU alerts
    if $snapshot.cpu.usage_percent > $config.alert_thresholds.cpu_usage_percent {
        let alert = {
            type: "high_cpu_usage",
            severity: "warning", 
            message: $"CPU usage at ($snapshot.cpu.usage_percent)%",
            threshold: $config.alert_thresholds.cpu_usage_percent,
            current_value: $snapshot.cpu.usage_percent,
            timestamp: $snapshot.timestamp,
            recommendations: [
                "Check for runaway processes",
                "Consider parallel vs sequential build tasks",
                "Monitor background processes"
            ]
        }
        $alerts | append $alert
        print $"🔥 CPU Alert: ($alert.message)"
    }
    
    # Disk space alerts
    let high_disk_usage = ($snapshot.disk.filesystems | where usage_percent > $config.alert_thresholds.disk_usage_percent)
    if ($high_disk_usage | length) > 0 {
        $high_disk_usage | each { |disk|
            let alert = {
                type: "high_disk_usage",
                severity: "warning",
                message: $"Disk ($disk.mount) usage at ($disk.usage_percent)% (($disk.used_gb)GB used)",
                threshold: $config.alert_thresholds.disk_usage_percent,
                current_value: $disk.usage_percent,
                timestamp: $snapshot.timestamp,
                disk_mount: $disk.mount,
                recommendations: [
                    "Clean up temporary files and caches",
                    "Remove old build artifacts",
                    "Archive or delete unused files"
                ]
            }
            $alerts | append $alert
            print $"💾 Disk Alert: ($alert.message)"
        }
    }
    
    # Save alerts if any
    if ($alerts | length) > 0 {
        $alerts | each { |alert| $alert | to json --raw } | str join "\n" | save --append .performance/resource_alerts.jsonl
    }
}

# Monitor resource usage during command execution
def "resource monitor-command" [
    command: string,
    context: string = "command_execution",
    --interval: int = 5  # seconds between measurements
] {
    resource init
    
    print $"🔍 Monitoring resource usage for: ($command)"
    
    # Record baseline
    let baseline = (resource record $"($context)_baseline")
    
    # Start background monitoring
    let monitor_script = $"
        while (ps | where name =~ '($command)' | length) > 0 {
            nu -c 'use scripts/resource-monitor.nu; resource record \"($context)_during\"'
            sleep ($interval)sec
        }
    "
    
    # Execute command and monitor
    let start_time = (date now)
    
    # Start monitoring in background
    nu -c $monitor_script &
    
    # Execute the actual command
    let result = (try {
        nu -c $command | complete
    } catch {
        {stdout: "", stderr: $in, exit_code: 1}
    })
    
    let end_time = (date now)
    let duration = $end_time - $start_time
    
    # Record final state
    let final_snapshot = (resource record $"($context)_final")
    
    # Generate monitoring report
    print $"✅ Command completed in ($duration)"
    print $"📊 Resource Usage Summary:"
    print $"   Memory: ($baseline.memory.usage_percent)% → ($final_snapshot.memory.usage_percent)%"
    print $"   CPU: ($baseline.cpu.usage_percent)% average during execution"
    
    if $final_snapshot.memory.usage_percent > ($baseline.memory.usage_percent + 10) {
        print $"⚠️  Significant memory increase detected (+{($final_snapshot.memory.usage_percent - $baseline.memory.usage_percent)}%)"
    }
    
    $result
}

# Generate resource usage report
def "resource report" [
    --hours: int = 24,
    --context: string = "all",
    --format: string = "summary"
] {
    resource init
    
    let start_time = (date now) - ($hours * 1hr)
    
    let usage_data = (
        open .performance/resource_usage.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $start_time
    )
    
    let filtered_data = if $context == "all" {
        $usage_data
    } else {
        $usage_data | where context == $context
    }
    
    if ($filtered_data | length) == 0 {
        print $"No resource usage data found for the last ($hours) hours"
        return
    }
    
    match $format {
        "summary" => {
            print $"📊 Resource Usage Report - Last ($hours) hours"
            print "=" * 50
            
            let memory_stats = ($filtered_data | get memory.usage_percent)
            let cpu_stats = ($filtered_data | get cpu.usage_percent)
            
            print $"Memory Usage:"
            print $"  Average: ($memory_stats | math avg | math round --precision 1)%"
            print $"  Peak: ($memory_stats | math max)%"
            print $"  Minimum: ($memory_stats | math min)%"
            
            print $"\nCPU Usage:"
            print $"  Average: ($cpu_stats | math avg | math round --precision 1)%"
            print $"  Peak: ($cpu_stats | math max)%"
            print $"  Minimum: ($cpu_stats | math min)%"
            
            # Context breakdown
            let context_summary = (
                $filtered_data 
                | group-by context
                | transpose context measurements
                | each { |row|
                    let data = $row.measurements
                    {
                        context: $row.context,
                        count: ($data | length),
                        avg_memory: ($data | get memory.usage_percent | math avg | math round --precision 1),
                        avg_cpu: ($data | get cpu.usage_percent | math avg | math round --precision 1),
                        peak_memory: ($data | get memory.usage_percent | math max)
                    }
                }
                | sort-by avg_memory --reverse
            )
            
            print $"\nUsage by Context:"
            $context_summary | table
            
            # Alert summary
            if (".performance/resource_alerts.jsonl" | path exists) {
                let recent_alerts = (
                    open .performance/resource_alerts.jsonl
                    | lines
                    | each { |line| $line | from json }
                    | where timestamp > $start_time
                    | group-by type
                    | transpose alert_type count
                    | each { |row| 
                        {
                            alert_type: $row.alert_type, 
                            count: ($row.count | length)
                        }
                    }
                )
                
                if ($recent_alerts | length) > 0 {
                    print $"\n⚠️  Recent Alerts:"
                    $recent_alerts | table
                }
            }
        },
        "detailed" => {
            $filtered_data | table
        },
        "json" => {
            $filtered_data | to json
        },
        _ => {
            print "Invalid format. Use: summary, detailed, json"
        }
    }
}

# Clean old resource usage data
def "resource cleanup" [--hours: int = 72] {
    resource init
    
    let cutoff_time = (date now) - ($hours * 1hr)
    
    # Clean usage data
    let current_usage = (
        open .performance/resource_usage.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $cutoff_time
    )
    
    $current_usage | each { |record| $record | to json --raw } | str join "\n" | save .performance/resource_usage.jsonl
    
    # Clean alerts
    if (".performance/resource_alerts.jsonl" | path exists) {
        let current_alerts = (
            open .performance/resource_alerts.jsonl
            | lines
            | each { |line| $line | from json }
            | where timestamp > $cutoff_time
        )
        
        $current_alerts | each { |alert| $alert | to json --raw } | str join "\n" | save .performance/resource_alerts.jsonl
    }
    
    print $"🧹 Cleaned resource data older than ($hours) hours"
}

# Start continuous resource monitoring
def "resource watch" [
    --interval: int = 30,  # seconds between measurements
    --context: string = "continuous_monitoring"
] {
    resource init
    
    print $"🔄 Starting continuous resource monitoring (interval: ($interval)s)"
    print "Press Ctrl+C to stop"
    
    while true {
        resource record $context
        sleep ($interval)sec
    }
}

# Resource usage optimization suggestions
def "resource optimize" [] {
    resource init
    
    let recent_data = (
        open .performance/resource_usage.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > ((date now) - 24hr)
    )
    
    if ($recent_data | length) == 0 {
        print "No recent resource data available for optimization analysis"
        return
    }
    
    print "🚀 Resource Optimization Suggestions"
    print "===================================="
    
    let avg_memory = ($recent_data | get memory.usage_percent | math avg)
    let max_memory = ($recent_data | get memory.usage_percent | math max)
    let avg_cpu = ($recent_data | get cpu.usage_percent | math avg)
    
    # Memory optimization suggestions
    if $avg_memory > 70 {
        print $"\n💾 Memory Optimization (Average: ($avg_memory | math round --precision 1)%):"
        print "• Consider closing unused applications and browser tabs"
        print "• Review memory-intensive development tools"
        print "• Check for memory leaks in running processes"
        
        if $max_memory > 90 {
            print "• Critical: Peak usage reached ($max_memory)% - consider upgrading RAM"
        }
    }
    
    # CPU optimization suggestions  
    if $avg_cpu > 60 {
        print $"\n🔥 CPU Optimization (Average: ($avg_cpu | math round --precision 1)%):"
        print "• Review background processes and services"
        print "• Consider using parallel builds with limited concurrency"
        print "• Check for runaway processes or infinite loops"
    }
    
    # Context-specific suggestions
    let context_analysis = (
        $recent_data
        | group-by context  
        | transpose context data
        | each { |row|
            let avg_mem = ($row.data | get memory.usage_percent | math avg)
            {context: $row.context, avg_memory: $avg_mem, count: ($row.data | length)}
        }
        | where avg_memory > 60
        | sort-by avg_memory --reverse
    )
    
    if ($context_analysis | length) > 0 {
        print $"\n🎯 High Resource Usage Contexts:"
        $context_analysis | each { |ctx|
            print $"• ($ctx.context): ($ctx.avg_memory | math round --precision 1)% memory average"
        }
    }
    
    # Disk space suggestions
    let recent_snapshot = ($recent_data | last)
    let high_disk_usage = ($recent_snapshot.disk.filesystems | where usage_percent > 80)
    
    if ($high_disk_usage | length) > 0 {
        print $"\n💿 Disk Space Optimization:"
        $high_disk_usage | each { |disk|
            print $"• ($disk.mount): ($disk.usage_percent)% used - Clean up temporary files"
        }
        print "• Run: devbox run clean in each environment"
        print "• Clear npm/cargo/pip caches"
        print "• Remove old Docker images and containers"
    }
}

# Main command dispatcher
def main [command: string, ...args] {
    match $command {
        "init" => { resource init },
        "snapshot" => { resource snapshot },
        "record" => { resource record ...$args },
        "monitor-command" => {
            if ($args | length) >= 1 {
                resource monitor-command $args.0 ...(($args | skip 1))
            } else {
                print "Usage: resource monitor-command <command> [context]"
            }
        },
        "report" => { resource report ...$args },
        "cleanup" => { resource cleanup ...$args },
        "watch" => { resource watch ...$args },
        "optimize" => { resource optimize },
        _ => {
            print "Resource Usage Monitoring System"
            print "Usage:"
            print "  resource init                     - Initialize resource monitoring"
            print "  resource snapshot [context]       - Take current resource snapshot"
            print "  resource record [context]         - Record resource usage with alerts"
            print "  resource monitor-command <cmd>    - Monitor resources during command execution"
            print "  resource report [--hours N]       - Generate resource usage report"
            print "  resource cleanup [--hours N]      - Clean old resource data"
            print "  resource watch [--interval N]     - Continuous resource monitoring"
            print "  resource optimize                 - Get resource optimization suggestions"
        }
    }
}