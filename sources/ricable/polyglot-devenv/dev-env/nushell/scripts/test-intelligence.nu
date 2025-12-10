#!/usr/bin/env nu
# Test Performance Intelligence and Flaky Test Detection System
# Analyzes test execution patterns, identifies flaky tests, and provides insights

# Initialize test intelligence
def "test-intel init" [] {
    mkdir .testing
    
    if not (".testing/test_results.jsonl" | path exists) {
        [] | save .testing/test_results.jsonl
    }
    
    if not (".testing/config.json" | path exists) {
        {
            version: "1.0",
            environments: {
                python: {
                    path: "python-env",
                    test_command: "devbox run test",
                    test_patterns: ["test_*.py", "*_test.py", "tests/*.py"],
                    output_parsers: {
                        pytest: {
                            pass_pattern: "PASSED",
                            fail_pattern: "FAILED",
                            skip_pattern: "SKIPPED",
                            duration_pattern: "([0-9.]+)s",
                            summary_pattern: "([0-9]+) passed.*in ([0-9.]+)s"
                        }
                    }
                },
                typescript: {
                    path: "typescript-env",
                    test_command: "devbox run test",
                    test_patterns: ["*.test.ts", "*.test.js", "*.spec.ts", "*.spec.js"],
                    output_parsers: {
                        jest: {
                            pass_pattern: "PASS",
                            fail_pattern: "FAIL",
                            skip_pattern: "SKIP",
                            duration_pattern: "([0-9.]+) ms",
                            summary_pattern: "Tests:\\s+([0-9]+) passed.*Time:\\s+([0-9.]+) s"
                        }
                    }
                },
                rust: {
                    path: "rust-env", 
                    test_command: "devbox run test",
                    test_patterns: ["*_test.rs", "tests/*.rs"],
                    output_parsers: {
                        cargo: {
                            pass_pattern: "test .* \\.\\.\\. ok",
                            fail_pattern: "test .* \\.\\.\\. FAILED",
                            skip_pattern: "test .* \\.\\.\\. ignored",
                            duration_pattern: "finished in ([0-9.]+)s",
                            summary_pattern: "([0-9]+) passed.*in ([0-9.]+)s"
                        }
                    }
                },
                go: {
                    path: "go-env",
                    test_command: "devbox run test", 
                    test_patterns: ["*_test.go"],
                    output_parsers: {
                        go: {
                            pass_pattern: "PASS",
                            fail_pattern: "FAIL",
                            skip_pattern: "SKIP",
                            duration_pattern: "([0-9.]+)s",
                            summary_pattern: "PASS.*([0-9.]+)s"
                        }
                    }
                }
            },
            flaky_detection: {
                min_runs: 5,                    # Minimum runs to detect flakiness
                failure_rate_threshold: 0.2,   # 20% failure rate = flaky
                inconsistent_duration_threshold: 2.0,  # 2x duration variance = potentially flaky
                analysis_window_days: 7
            },
            performance_thresholds: {
                slow_test_threshold_seconds: 5.0,
                very_slow_test_threshold_seconds: 30.0,
                suite_duration_warning_seconds: 300.0,
                degradation_threshold_percent: 50.0
            },
            retention_days: 30
        } | save .testing/config.json
    }
}

# Parse test output based on environment and test runner
def "test-intel parse-output" [
    output: string,
    environment: string,
    test_runner: string
] {
    let config = (open .testing/config.json)
    let env_config = ($config.environments | get $environment)
    
    if not ($test_runner in $env_config.output_parsers) {
        return {
            parsed: false,
            reason: $"Unknown test runner: ($test_runner)",
            raw_output: $output
        }
    }
    
    let parser = ($env_config.output_parsers | get $test_runner)
    let lines = ($output | lines)
    
    mut test_results = []
    mut summary = {}
    
    # Parse individual test results
    for line in $lines {
        # Check for passed tests
        if ($line | str contains $parser.pass_pattern) {
            let test_name = ($line | parse --regex ".*test ([^\\s]+).*" | get capture0? | first | default "unknown")
            let duration = (try {
                $line | parse --regex $parser.duration_pattern | get capture0? | first | into float
            } catch { 0.0 })
            
            $test_results = ($test_results | append {
                name: $test_name,
                status: "passed",
                duration_seconds: $duration,
                line: $line
            })
        }
        
        # Check for failed tests
        if ($line | str contains $parser.fail_pattern) {
            let test_name = ($line | parse --regex ".*test ([^\\s]+).*" | get capture0? | first | default "unknown")
            
            $test_results = ($test_results | append {
                name: $test_name,
                status: "failed", 
                duration_seconds: 0.0,
                line: $line
            })
        }
        
        # Check for skipped tests
        if ($line | str contains $parser.skip_pattern) {
            let test_name = ($line | parse --regex ".*test ([^\\s]+).*" | get capture0? | first | default "unknown")
            
            $test_results = ($test_results | append {
                name: $test_name,
                status: "skipped",
                duration_seconds: 0.0,
                line: $line
            })
        }
        
        # Parse summary information
        if ($line | str contains $parser.summary_pattern) {
            $summary = (try {
                let parsed = ($line | parse --regex $parser.summary_pattern)
                {
                    total_passed: ($parsed | get capture0? | first | into int),
                    total_duration: ($parsed | get capture1? | first | into float)
                }
            } catch {
                { total_passed: 0, total_duration: 0.0 }
            })
        }
    }
    
    {
        parsed: true,
        test_results: $test_results,
        summary: $summary,
        total_tests: ($test_results | length),
        passed_tests: ($test_results | where status == "passed" | length),
        failed_tests: ($test_results | where status == "failed" | length),
        skipped_tests: ($test_results | where status == "skipped" | length)
    }
}

# Run tests with intelligence collection
def "test-intel run" [
    environment: string,
    --test-runner: string = "auto",
    --collect-coverage = false
] {
    test-intel init
    
    let config = (open .testing/config.json)
    
    if not ($environment in $config.environments) {
        print $"❌ Unknown environment: ($environment)"
        return
    }
    
    let env_config = ($config.environments | get $environment)
    let env_path = $env_config.path
    
    if not ($env_path | path exists) {
        print $"❌ Environment path not found: ($env_path)"
        return
    }
    
    print $"🧪 Running intelligent test analysis for ($environment)..."
    
    cd $env_path
    
    # Record pre-test state
    let pre_test_snapshot = {
        timestamp: (date now),
        memory_usage: (sys | get host.memory.used),
        cpu_usage: (sys | get host.cpu | each { |core| $core.cpu_usage } | math avg)
    }
    
    # Execute tests with timing
    let start_time = (date now)
    
    let test_result = (try {
        nu -c $env_config.test_command | complete
    } catch {
        {stdout: "", stderr: $in, exit_code: 1}
    })
    
    let end_time = (date now)
    let total_duration = ($end_time - $start_time)
    
    # Record post-test state
    let post_test_snapshot = {
        timestamp: (date now),
        memory_usage: (sys | get host.memory.used),
        cpu_usage: (sys | get host.cpu | each { |core| $core.cpu_usage } | math avg)
    }
    
    cd ..
    
    # Auto-detect test runner if not specified
    let detected_runner = if $test_runner == "auto" {
        if ($test_result.stdout | str contains "pytest") { "pytest" }
        else if ($test_result.stdout | str contains "jest") { "jest" }
        else if ($test_result.stdout | str contains "cargo test") { "cargo" }
        else if ($test_result.stdout | str contains "go test") { "go" }
        else { "unknown" }
    } else { $test_runner }
    
    # Parse test output
    let parsed_output = (test-intel parse-output $test_result.stdout $environment $detected_runner)
    
    # Collect additional metrics
    let resource_delta = {
        memory_delta_mb: (($post_test_snapshot.memory_usage - $pre_test_snapshot.memory_usage) / 1MB),
        cpu_average: (($pre_test_snapshot.cpu_usage + $post_test_snapshot.cpu_usage) / 2)
    }
    
    # Create comprehensive test intelligence record
    let intelligence_record = {
        timestamp: $start_time,
        environment: $environment,
        test_runner: $detected_runner,
        total_duration_seconds: ($total_duration / 1sec),
        exit_code: $test_result.exit_code,
        success: ($test_result.exit_code == 0),
        parsed_results: $parsed_output,
        resource_usage: $resource_delta,
        raw_output: {
            stdout: $test_result.stdout,
            stderr: $test_result.stderr
        },
        system_context: {
            pre_test: $pre_test_snapshot,
            post_test: $post_test_snapshot
        }
    }
    
    # Save test intelligence
    $intelligence_record | to json --raw | save --append .testing/test_results.jsonl
    
    # Analyze for immediate insights
    test-intel analyze-run $intelligence_record
    
    print $"✅ Test intelligence collected and analyzed"
    $intelligence_record
}

# Analyze a single test run for immediate insights
def "test-intel analyze-run" [record: record] {
    let config = (open .testing/config.json)
    let thresholds = $config.performance_thresholds
    
    print $"📊 Test Run Analysis:"
    print $"   Duration: ($record.total_duration_seconds)s"
    print $"   Success: ($record.success)"
    
    if $record.parsed_results.parsed {
        let results = $record.parsed_results
        print $"   Tests: ($results.total_tests) total, ($results.passed_tests) passed, ($results.failed_tests) failed"
        
        # Check for slow tests
        if $results.total_tests > 0 {
            let slow_tests = ($results.test_results | where duration_seconds > $thresholds.slow_test_threshold_seconds)
            if ($slow_tests | length) > 0 {
                print $"   ⚠️  Slow tests detected: ($slow_tests | length)"
                $slow_tests | each { |test|
                    print $"      • ($test.name): ($test.duration_seconds)s"
                }
            }
        }
        
        # Check overall suite performance
        if $record.total_duration_seconds > $thresholds.suite_duration_warning_seconds {
            print $"   🐌 Test suite is slow (>($thresholds.suite_duration_warning_seconds)s)"
        }
    }
    
    # Resource usage analysis
    if $record.resource_usage.memory_delta_mb > 100 {
        print $"   💾 High memory usage: ($record.resource_usage.memory_delta_mb)MB"
    }
    
    if $record.resource_usage.cpu_average > 80 {
        print $"   🔥 High CPU usage: ($record.resource_usage.cpu_average)%"
    }
}

# Detect flaky tests based on historical data
def "test-intel detect-flaky" [
    --environment: string = "all",
    --days: int = 7,
    --min-runs: int = 5
] {
    test-intel init
    
    let config = (open .testing/config.json)
    let start_date = (date now) - ($days * 1day)
    
    let test_data = (
        open .testing/test_results.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $start_date
        | where parsed_results.parsed == true
    )
    
    let filtered_data = if $environment == "all" {
        $test_data
    } else {
        $test_data | where environment == $environment
    }
    
    if ($filtered_data | length) == 0 {
        print "No test data found for flaky test analysis"
        return []
    }
    
    print $"🔍 Analyzing ($filtered_data | length) test runs for flaky tests..."
    
    # Extract all individual test results
    let all_test_results = (
        $filtered_data 
        | each { |run|
            $run.parsed_results.test_results | each { |test|
                $test | insert run_timestamp $run.timestamp | insert environment $run.environment
            }
        }
        | flatten
    )
    
    # Group by test name and environment
    let test_groups = (
        $all_test_results
        | group-by { |test| $"($test.environment):($test.name)" }
        | transpose test_key results
    )
    
    mut flaky_tests = []
    
    for group in $test_groups {
        let test_results = $group.results
        let test_runs = ($test_results | length)
        
        if $test_runs < $min_runs {
            continue
        }
        
        let test_name = ($test_results | first | get name)
        let environment = ($test_results | first | get environment)
        
        # Calculate failure rate
        let failures = ($test_results | where status == "failed" | length)
        let failure_rate = ($failures / $test_runs)
        
        # Calculate duration variance for passed tests
        let passed_durations = ($test_results | where status == "passed" | get duration_seconds | where $it > 0)
        let duration_variance = if ($passed_durations | length) > 1 {
            let mean = ($passed_durations | math avg)
            let variance = ($passed_durations | each { |d| ($d - $mean) ** 2 } | math avg)
            ($variance | math sqrt) / $mean
        } else { 0.0 }
        
        # Check for flaky behavior patterns
        let is_flaky = (
            ($failure_rate > 0 and $failure_rate < 1.0 and $failure_rate >= $config.flaky_detection.failure_rate_threshold) or
            ($duration_variance > $config.flaky_detection.inconsistent_duration_threshold)
        )
        
        if $is_flaky {
            let flaky_reason = if $failure_rate >= $config.flaky_detection.failure_rate_threshold {
                "intermittent_failures"
            } else {
                "inconsistent_duration"
            }
            
            $flaky_tests = ($flaky_tests | append {
                name: $test_name,
                environment: $environment,
                total_runs: $test_runs,
                failures: $failures,
                failure_rate: ($failure_rate * 100 | math round --precision 1),
                duration_variance: ($duration_variance | math round --precision 2),
                flaky_reason: $flaky_reason,
                severity: (if $failure_rate > 0.5 { "high" } else if $failure_rate > 0.2 { "medium" } else { "low" }),
                recent_runs: ($test_results | sort-by run_timestamp | last 5)
            })
        }
    }
    
    # Sort by severity and failure rate
    $flaky_tests = ($flaky_tests | sort-by failure_rate --reverse)
    
    if ($flaky_tests | length) > 0 {
        print $"🎯 Detected ($flaky_tests | length) flaky tests:"
        $flaky_tests | each { |test|
            let severity_emoji = match $test.severity {
                "high" => "🔴",
                "medium" => "🟡",
                "low" => "🟢"
            }
            print $"($severity_emoji) ($test.environment):($test.name) - ($test.failure_rate)% failure rate ($test.failures)/($test.total_runs) runs"
        }
    } else {
        print "✅ No flaky tests detected!"
    }
    
    $flaky_tests
}

# Analyze test performance trends
def "test-intel analyze-trends" [
    --environment: string = "all",
    --days: int = 14,
    --test-name: string = ""
] {
    test-intel init
    
    let start_date = (date now) - ($days * 1day)
    
    let test_data = (
        open .testing/test_results.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $start_date
        | where success == true
        | where parsed_results.parsed == true
    )
    
    let filtered_data = if $environment == "all" {
        $test_data
    } else {
        $test_data | where environment == $environment
    }
    
    if ($filtered_data | length) < 2 {
        print "Insufficient data for trend analysis (need at least 2 successful runs)"
        return
    }
    
    print $"📈 Test Performance Trend Analysis - Last ($days) days"
    print "=" * 55
    
    # Overall suite performance trends
    let suite_durations = ($filtered_data | get total_duration_seconds | sort)
    let first_half = ($suite_durations | first (($suite_durations | length) / 2 | into int))
    let second_half = ($suite_durations | last (($suite_durations | length) / 2 | into int))
    
    let first_avg = ($first_half | math avg)
    let second_avg = ($second_half | math avg)
    let trend_change = (($second_avg - $first_avg) / $first_avg * 100)
    
    let trend_emoji = if $trend_change > 10 { "📈🔴" } else if $trend_change < -10 { "📉🟢" } else { "➡️" }
    print $"Overall Suite Performance: ($trend_emoji) ($trend_change | math round --precision 1)% change"
    print $"  Average duration: ($first_avg | math round --precision 1)s → ($second_avg | math round --precision 1)s"
    
    # Environment-specific trends
    let env_trends = (
        $filtered_data
        | group-by environment
        | transpose environment data
        | each { |row|
            let sorted_data = ($row.data | sort-by timestamp)
            if ($sorted_data | length) > 1 {
                let durations = ($sorted_data | get total_duration_seconds)
                let first_duration = ($durations | first)
                let last_duration = ($durations | last)
                let change = (($last_duration - $first_duration) / $first_duration * 100)
                
                {
                    environment: $row.environment,
                    runs: ($sorted_data | length),
                    avg_duration: ($durations | math avg | math round --precision 1),
                    trend_change: ($change | math round --precision 1),
                    latest_duration: ($last_duration | math round --precision 1)
                }
            }
        }
        | compact
    )
    
    print $"\nEnvironment-Specific Trends:"
    $env_trends | each { |trend|
        let trend_emoji = if $trend.trend_change > 10 { "📈" } else if $trend.trend_change < -10 { "📉" } else { "➡️" }
        print $"  ($trend.environment): ($trend_emoji) ($trend.trend_change)% change, ($trend.runs) runs, avg ($trend.avg_duration)s"
    }
    
    # Individual test analysis (if specific test requested)
    if $test_name != "" {
        print $"\nIndividual Test Analysis: ($test_name)"
        let test_results = (
            $filtered_data
            | each { |run|
                $run.parsed_results.test_results 
                | where name == $test_name
                | each { |test| $test | insert run_timestamp $run.timestamp }
            }
            | flatten
            | sort-by run_timestamp
        )
        
        if ($test_results | length) > 0 {
            let durations = ($test_results | get duration_seconds | where $it > 0)
            if ($durations | length) > 1 {
                let avg_duration = ($durations | math avg)
                let max_duration = ($durations | math max)
                let min_duration = ($durations | math min)
                let variance = ($durations | math stddev)
                
                print $"  Runs: ($test_results | length)"
                print $"  Duration: avg ($avg_duration | math round --precision 2)s, min ($min_duration)s, max ($max_duration)s"
                print $"  Variance: ($variance | math round --precision 2)s"
                
                if $variance > ($avg_duration * 0.5) {
                    print $"  ⚠️  High duration variance - potentially flaky test"
                }
            }
        } else {
            print $"  No data found for test: ($test_name)"
        }
    }
    
    # Performance regression detection
    let recent_runs = ($filtered_data | sort-by timestamp | last 5)
    let older_runs = ($filtered_data | sort-by timestamp | first 5)
    
    if ($recent_runs | length) > 0 and ($older_runs | length) > 0 {
        let recent_avg = ($recent_runs | get total_duration_seconds | math avg)
        let older_avg = ($older_runs | get total_duration_seconds | math avg)
        let regression = (($recent_avg - $older_avg) / $older_avg * 100)
        
        if $regression > 20 {
            print $"\n🚨 Performance Regression Detected!"
            print $"   Recent average: ($recent_avg | math round --precision 1)s"
            print $"   Previous average: ($older_avg | math round --precision 1)s"
            print $"   Regression: ($regression | math round --precision 1)%"
        }
    }
}

# Generate comprehensive test intelligence report
def "test-intel report" [
    --environment: string = "all", 
    --days: int = 7,
    --format: string = "summary"
] {
    test-intel init
    
    let start_date = (date now) - ($days * 1day)
    
    let test_data = (
        open .testing/test_results.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $start_date
    )
    
    let filtered_data = if $environment == "all" {
        $test_data
    } else {
        $test_data | where environment == $environment
    }
    
    match $format {
        "summary" => {
            print $"🧪 Test Intelligence Report - Last ($days) days"
            print "=" * 50
            
            if ($filtered_data | length) == 0 {
                print "No test data available for the specified period"
                return
            }
            
            # Overall statistics
            let total_runs = ($filtered_data | length)
            let successful_runs = ($filtered_data | where success == true | length)
            let success_rate = ($successful_runs / $total_runs * 100)
            
            print $"Test Execution Summary:"
            print $"  Total test runs: ($total_runs)"
            print $"  Successful runs: ($successful_runs) (($success_rate | math round --precision 1)%)"
            
            # Performance summary
            let successful_data = ($filtered_data | where success == true)
            if ($successful_data | length) > 0 {
                let avg_duration = ($successful_data | get total_duration_seconds | math avg)
                let max_duration = ($successful_data | get total_duration_seconds | math max)
                
                print $"  Average duration: ($avg_duration | math round --precision 1)s"
                print $"  Longest run: ($max_duration | math round --precision 1)s"
            }
            
            # Environment breakdown
            let env_summary = (
                $filtered_data
                | group-by environment
                | transpose environment data
                | each { |row|
                    let successful = ($row.data | where success == true | length)
                    let total = ($row.data | length)
                    {
                        environment: $row.environment,
                        runs: $total,
                        success_rate: ($successful / $total * 100 | math round --precision 1),
                        avg_duration: (
                            $row.data 
                            | where success == true 
                            | get total_duration_seconds 
                            | math avg 
                            | math round --precision 1
                        )
                    }
                }
            )
            
            print $"\nEnvironment Summary:"
            $env_summary | table
            
            # Flaky test detection
            let flaky_tests = (test-intel detect-flaky --environment $environment --days $days)
            if ($flaky_tests | length) > 0 {
                print $"\n⚠️  Flaky Tests Detected: ($flaky_tests | length)"
                $flaky_tests | first 3 | each { |test|
                    print $"  • ($test.environment):($test.name) - ($test.failure_rate)% failure rate"
                }
            }
            
            # Recent performance issues
            let recent_failures = ($filtered_data | where success == false | sort-by timestamp | last 5)
            if ($recent_failures | length) > 0 {
                print $"\nRecent Test Failures: ($recent_failures | length)"
                $recent_failures | each { |failure|
                    print $"  • ($failure.environment) at (($failure.timestamp | format date '%m-%d %H:%M'))"
                }
            }
        },
        "detailed" => {
            $filtered_data | to json
        },
        "flaky" => {
            test-intel detect-flaky --environment $environment --days $days
        },
        "trends" => {
            test-intel analyze-trends --environment $environment --days $days
        },
        _ => {
            print "Invalid format. Use: summary, detailed, flaky, trends"
        }
    }
}

# Clean old test intelligence data
def "test-intel cleanup" [--days: int = 30] {
    test-intel init
    
    let cutoff_date = (date now) - ($days * 1day)
    
    let current_data = (
        open .testing/test_results.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $cutoff_date
    )
    
    $current_data | each { |record| $record | to json --raw } | str join "\n" | save .testing/test_results.jsonl
    
    print $"🧹 Cleaned test intelligence data older than ($days) days"
}

# Main command dispatcher
def main [command: string, ...args] {
    match $command {
        "init" => { test-intel init },
        "run" => { 
            if ($args | length) >= 1 {
                test-intel run $args.0
            } else {
                print "Usage: test-intel run <environment> [--test-runner runner]"
            }
        },
        "detect-flaky" => { test-intel detect-flaky },
        "analyze-trends" => { test-intel analyze-trends },
        "report" => { test-intel report },
        "cleanup" => { test-intel cleanup },
        _ => {
            print "Test Performance Intelligence and Flaky Test Detection System"
            print "Usage:"
            print "  test-intel init                   - Initialize test intelligence"
            print "  test-intel run <environment>      - Run tests with intelligence collection"
            print "  test-intel detect-flaky [--env]   - Detect flaky tests from historical data"
            print "  test-intel analyze-trends [--env] - Analyze test performance trends"
            print "  test-intel report [--format]      - Generate comprehensive test report"
            print "  test-intel cleanup [--days N]     - Clean old test intelligence data"
        }
    }
}