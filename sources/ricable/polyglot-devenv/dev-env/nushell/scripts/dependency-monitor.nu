#!/usr/bin/env nu
# Comprehensive Dependency Monitoring System for Polyglot Development Environment
# Monitors outdated dependencies, security vulnerabilities, and license compliance

source ../../devpod-automation/config.nu

# Initialize dependency monitoring
def "deps init" [] {
    if not (".dependencies" | path exists) {
        mkdir .dependencies
    }
    
    if not (".dependencies/scan_results.jsonl" | path exists) {
        [] | save .dependencies/scan_results.jsonl
    }
    
    if not (".dependencies/config.json" | path exists) {
        {
            version: "1.0",
            environments: {
                python: { path: "python-env", manager: "uv", manifest: "pyproject.toml" },
                typescript: { path: "typescript-env", manager: "npm", manifest: "package.json" },
                rust: { path: "rust-env", manager: "cargo", manifest: "Cargo.toml" },
                go: { path: "go-env", manager: "go", manifest: "go.mod" }
            },
            scan_schedule: "daily",
            alert_thresholds: {
                critical_vulnerabilities: 0,
                high_vulnerabilities: 3,
                outdated_packages_percentage: 30,
                license_violations: 0
            },
            retention_days: 90
        } | save .dependencies/config.json
    }
}

# Scan Python dependencies using uv
def "deps scan-python" [] {
    let env_path = $"($config.output_dir)/python-env"
    
    if not ($env_path | path exists) {
        return {
            environment: "python",
            status: "error",
            message: "Python environment not found",
            timestamp: (date now)
        }
    }
    
    cd $env_path
    
    # Get current dependencies
    let current_deps = (try {
        uv pip list --format json | from json
    } catch {
        []
    })
    
    # Check for outdated packages
    let outdated = (try {
        uv pip list --outdated --format json | from json
    } catch {
        []
    })
    
    # Security audit (if available)
    let security_issues = (try {
        uv pip audit --format json | from json
    } catch {
        []
    })
    
    # License information (basic)
    let license_info = (try {
        $current_deps | each { |pkg|
            let license_check = (try {
                uv pip show $pkg.name | lines | where ($it | str contains "License:") | first | split column ":" | get column1? | default "Unknown"
            } catch {
                "Unknown"
            })
            
            {
                name: $pkg.name,
                version: $pkg.version,
                license: $license_check
            }
        }
    } catch {
        []
    })
    
    cd ..
    
    {
        environment: "python",
        timestamp: (date now),
        status: "success",
        total_packages: ($current_deps | length),
        outdated_packages: ($outdated | length),
        outdated_percentage: (if ($current_deps | length) > 0 { ($outdated | length) / ($current_deps | length) * 100 | math round --precision 1 } else { 0 }),
        security_issues: ($security_issues | length),
        packages: $current_deps,
        outdated: $outdated,
        vulnerabilities: $security_issues,
        licenses: $license_info
    }
}

# Scan TypeScript dependencies using npm
def "deps scan-typescript" [] {
    let env_path = $"($config.output_dir)/typescript-env"
    
    if not ($env_path | path exists) {
        return {
            environment: "typescript",
            status: "error", 
            message: "TypeScript environment not found",
            timestamp: (date now)
        }
    }
    
    cd $env_path
    
    # Get current dependencies
    let current_deps = (try {
        npm list --json | from json | get dependencies? | default {} | transpose name info | each { |pkg|
            {
                name: $pkg.name,
                version: $pkg.info.version,
                resolved: ($pkg.info.resolved? | default ""),
                required: ($pkg.info.required? | default "")
            }
        }
    } catch {
        []
    })
    
    # Check for outdated packages
    let outdated = (try {
        npm outdated --json | from json | transpose name info | each { |pkg|
            {
                name: $pkg.name,
                current: $pkg.info.current,
                wanted: $pkg.info.wanted,
                latest: $pkg.info.latest,
                location: ($pkg.info.location? | default "")
            }
        }
    } catch {
        []
    })
    
    # Security audit
    let security_audit = (try {
        npm audit --json | from json
    } catch {
        { vulnerabilities: {} }
    })
    
    let vulnerabilities = (try {
        $security_audit.vulnerabilities? | default {} | transpose name info | each { |vuln|
            {
                name: $vuln.name,
                severity: ($vuln.info.severity? | default "unknown"),
                title: ($vuln.info.title? | default ""),
                url: ($vuln.info.url? | default "")
            }
        }
    } catch {
        []
    })
    
    # License information
    let license_info = (try {
        npm ls --json | from json | get dependencies? | default {} | transpose name info | each { |pkg|
            {
                name: $pkg.name,
                version: ($pkg.info.version? | default ""),
                license: (try {
                    npm view $pkg.name license 2>/dev/null
                } catch {
                    "Unknown"
                })
            }
        }
    } catch {
        []
    })
    
    cd ..
    
    {
        environment: "typescript",
        timestamp: (date now),
        status: "success",
        total_packages: ($current_deps | length),
        outdated_packages: ($outdated | length),
        outdated_percentage: (if ($current_deps | length) > 0 { ($outdated | length) / ($current_deps | length) * 100 | math round --precision 1 } else { 0 }),
        security_issues: ($vulnerabilities | length),
        packages: $current_deps,
        outdated: $outdated,
        vulnerabilities: $vulnerabilities,
        licenses: $license_info
    }
}

# Scan Rust dependencies using cargo
def "deps scan-rust" [] {
    let env_path = $"($config.output_dir)/rust-env"
    
    if not ($env_path | path exists) {
        return {
            environment: "rust",
            status: "error",
            message: "Rust environment not found", 
            timestamp: (date now)
        }
    }
    
    cd $env_path
    
    # Get current dependencies
    let current_deps = (try {
        cargo tree --format "{p} {l}" --prefix none | lines | each { |line|
            let parts = ($line | split column " " name license)
            {
                name: ($parts | get name | split column ":" | get column0),
                version: ($parts | get name | split column ":" | get column1? | default ""),
                license: ($parts | get license? | default "Unknown")
            }
        } | uniq-by name
    } catch {
        []
    })
    
    # Check for outdated packages
    let outdated = (try {
        cargo outdated --format json | from json | get dependencies? | default []
    } catch {
        []
    })
    
    # Security audit using cargo-audit
    let security_issues = (try {
        cargo audit --json | from json | get vulnerabilities? | default []
    } catch {
        []
    })
    
    cd ..
    
    {
        environment: "rust",
        timestamp: (date now),
        status: "success",
        total_packages: ($current_deps | length),
        outdated_packages: ($outdated | length),
        outdated_percentage: (if ($current_deps | length) > 0 { ($outdated | length) / ($current_deps | length) * 100 | math round --precision 1 } else { 0 }),
        security_issues: ($security_issues | length),
        packages: $current_deps,
        outdated: $outdated,
        vulnerabilities: $security_issues,
        licenses: ($current_deps | get license)
    }
}

# Scan Go dependencies
def "deps scan-go" [] {
    let env_path = $"($config.output_dir)/go-env"
    
    if not ($env_path | path exists) {
        return {
            environment: "go",
            status: "error",
            message: "Go environment not found",
            timestamp: (date now)
        }
    }
    
    cd $env_path
    
    # Get current dependencies
    let current_deps = (try {
        go list -m -json all | lines | each { |line| 
            if ($line | str trim | str length) > 0 {
                $line | from json
            }
        } | compact | each { |dep|
            {
                name: ($dep.Path? | default ""),
                version: ($dep.Version? | default ""),
                indirect: ($dep.Indirect? | default false)
            }
        }
    } catch {
        []
    })
    
    # Check for outdated packages
    let outdated = (try {
        go list -m -u -json all | lines | each { |line|
            if ($line | str trim | str length) > 0 {
                $line | from json
            }
        } | compact | where Update? != null | each { |dep|
            {
                name: ($dep.Path? | default ""),
                current: ($dep.Version? | default ""),
                available: ($dep.Update.Version? | default "")
            }
        }
    } catch {
        []
    })
    
    # Security vulnerabilities using govulncheck
    let security_issues = (try {
        govulncheck -json ./... | lines | each { |line|
            if ($line | str trim | str length) > 0 {
                $line | from json
            }
        } | compact | where message.level? == "VULN"
    } catch {
        []
    })
    
    cd ..
    
    {
        environment: "go",
        timestamp: (date now),
        status: "success",
        total_packages: ($current_deps | length),
        outdated_packages: ($outdated | length),
        outdated_percentage: (if ($current_deps | length) > 0 { ($outdated | length) / ($current_deps | length) * 100 | math round --precision 1 } else { 0 }),
        security_issues: ($security_issues | length),
        packages: $current_deps,
        outdated: $outdated,
        vulnerabilities: $security_issues,
        licenses: []
    }
}

# Scan all environments
def "deps scan-all" [] {
    deps init
    
    print "🔍 Scanning dependencies across all environments..."
    
    let python_result = (deps scan-python)
    let typescript_result = (deps scan-typescript)
    let rust_result = (deps scan-rust)
    let go_result = (deps scan-go)
    
    let results = [$python_result, $typescript_result, $rust_result, $go_result]
    
    # Save scan results
    $results | each { |result| 
        $result | to json --raw | save --append .dependencies/scan_results.jsonl
    }
    
    # Check for alerts
    deps check-alerts $results
    
    print "✅ Dependency scan completed"
    $results
}

# Check for dependency alerts
def "deps check-alerts" [scan_results: list] {
    let config = (open .dependencies/config.json)
    let alerts = []
    
    for result in $scan_results {
        if $result.status != "success" { continue }
        
        let env_name = $result.environment
        
        # Critical vulnerabilities alert
        if $result.security_issues > $config.alert_thresholds.critical_vulnerabilities {
            let alert = {
                type: "critical_vulnerabilities",
                environment: $env_name,
                severity: "critical",
                count: $result.security_issues,
                message: $"($env_name): ($result.security_issues) security vulnerabilities found",
                timestamp: (date now),
                recommendations: [
                    "Review and update vulnerable packages immediately",
                    "Check for security patches",
                    "Consider alternative packages if fixes unavailable"
                ]
            }
            $alerts | append $alert
            print $"🚨 Critical Security Alert: ($alert.message)"
        }
        
        # Outdated packages alert
        if $result.outdated_percentage > $config.alert_thresholds.outdated_packages_percentage {
            let alert = {
                type: "outdated_packages",
                environment: $env_name,
                severity: "warning",
                percentage: $result.outdated_percentage,
                count: $result.outdated_packages,
                message: $"($env_name): ($result.outdated_percentage)% of packages are outdated (($result.outdated_packages)/($result.total_packages))",
                timestamp: (date now),
                recommendations: [
                    "Review and update outdated packages",
                    "Test thoroughly after updates",
                    "Consider automated dependency updates"
                ]
            }
            $alerts | append $alert
            print $"⚠️  Outdated Dependencies Alert: ($alert.message)"
        }
    }
    
    # Save alerts if any
    if ($alerts | length) > 0 {
        $alerts | each { |alert| $alert | to json --raw } | str join "\n" | save --append .dependencies/alerts.jsonl
    }
}

# Generate dependency report
def "deps report" [
    --days: int = 7,
    --environment: string = "all",
    --format: string = "summary"
] {
    deps init
    
    let start_date = (date now) - ($days * 1day)
    
    let scan_data = (
        open .dependencies/scan_results.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $start_date
    )
    
    let filtered_data = if $environment == "all" {
        $scan_data
    } else {
        $scan_data | where environment == $environment
    }
    
    if ($filtered_data | length) == 0 {
        print $"No dependency scan data found for the last ($days) days"
        return
    }
    
    match $format {
        "summary" => {
            print $"📦 Dependency Health Report - Last ($days) days"
            print "=" * 55
            
            # Get latest scan for each environment
            let latest_scans = (
                $filtered_data
                | group-by environment
                | transpose environment scans
                | each { |row|
                    $row.scans | sort-by timestamp | last
                }
            )
            
            print "Current Status:"
            $latest_scans | each { |scan|
                let status_emoji = if $scan.security_issues > 0 { "🚨" } else if $scan.outdated_percentage > 30 { "⚠️" } else { "✅" }
                print $"($status_emoji) ($scan.environment): ($scan.total_packages) packages, ($scan.outdated_packages) outdated (($scan.outdated_percentage)%), ($scan.security_issues) vulnerabilities"
            }
            
            # Trend analysis
            print "\nTrend Analysis:"
            let trends = (
                $filtered_data
                | group-by environment
                | transpose environment scans
                | each { |row|
                    let sorted_scans = ($row.scans | sort-by timestamp)
                    if ($sorted_scans | length) > 1 {
                        let first = ($sorted_scans | first)
                        let last = ($sorted_scans | last)
                        
                        {
                            environment: $row.environment,
                            package_change: ($last.total_packages - $first.total_packages),
                            outdated_change: ($last.outdated_packages - $first.outdated_packages),
                            security_change: ($last.security_issues - $first.security_issues)
                        }
                    }
                }
                | compact
            )
            
            $trends | each { |trend|
                let package_emoji = if $trend.package_change > 0 { "📈" } else if $trend.package_change < 0 { "📉" } else { "➡️" }
                let security_emoji = if $trend.security_change > 0 { "⬆️🚨" } else if $trend.security_change < 0 { "⬇️✅" } else { "➡️" }
                print $"  ($trend.environment): Packages ($package_emoji) ($trend.package_change), Security ($security_emoji) ($trend.security_change)"
            }
            
            # Recent alerts
            if (".dependencies/alerts.jsonl" | path exists) {
                let recent_alerts = (
                    open .dependencies/alerts.jsonl
                    | lines
                    | each { |line| $line | from json }
                    | where timestamp > $start_date
                )
                
                if ($recent_alerts | length) > 0 {
                    print $"\nRecent Alerts: ($recent_alerts | length)"
                    $recent_alerts | each { |alert|
                        print $"  • ($alert.type): ($alert.message)"
                    }
                }
            }
        },
        "detailed" => {
            $filtered_data | table
        },
        "vulnerabilities" => {
            let vuln_data = (
                $filtered_data
                | where security_issues > 0
                | each { |scan|
                    $scan.vulnerabilities | each { |vuln|
                        $vuln | insert environment $scan.environment | insert scan_date $scan.timestamp
                    }
                }
                | flatten
            )
            
            if ($vuln_data | length) > 0 {
                print "🚨 Security Vulnerabilities:"
                $vuln_data | group-by severity | transpose severity vulns | each { |row|
                    print $"\n($row.severity | str upcase):"
                    $row.vulns | each { |vuln|
                        print $"  • ($vuln.environment): ($vuln.name) - ($vuln.title? | default 'No description')"
                    }
                }
            } else {
                print "✅ No security vulnerabilities found!"
            }
        },
        "json" => {
            $filtered_data | to json
        },
        _ => {
            print "Invalid format. Use: summary, detailed, vulnerabilities, json"
        }
    }
}

# Update dependencies interactively
def "deps update" [
    environment?: string,
    --auto = false,  # Automatically update without prompts
    --security-only = false  # Only update packages with security issues
] {
    let environments_to_update = if $environment != null {
        [$environment]
    } else {
        ["python", "typescript", "rust", "go"]
    }
    
    for env in $environments_to_update {
        print $"🔄 Updating ($env) dependencies..."
        
        # Get latest scan for environment
        let latest_scan = (
            open .dependencies/scan_results.jsonl
            | lines
            | each { |line| $line | from json }
            | where environment == $env
            | sort-by timestamp
            | last
        )
        
        if $latest_scan == null {
            print $"No scan data available for ($env). Run 'deps scan-all' first."
            continue
        }
        
        let packages_to_update = if $security_only {
            # Only update packages with security vulnerabilities
            $latest_scan.vulnerabilities | get name | uniq
        } else {
            # Update all outdated packages
            $latest_scan.outdated | get name
        }
        
        if ($packages_to_update | length) == 0 {
            print $"✅ No updates needed for ($env)"
            continue
        }
        
        print $"Found ($packages_to_update | length) packages to update:"
        $packages_to_update | each { |pkg| print $"  • ($pkg)" }
        
        if not $auto {
            let confirm = (input $"Proceed with updating ($env) dependencies? (y/n): ")
            if $confirm != "y" and $confirm != "yes" {
                print "Skipping update."
                continue
            }
        }
        
        # Perform updates based on environment
        match $env {
            "python" => {
                cd python-env
                for pkg in $packages_to_update {
                    print $"Updating ($pkg)..."
                    uv add $"($pkg)@latest" --upgrade
                }
                cd ..
            },
            "typescript" => {
                cd typescript-env
                for pkg in $packages_to_update {
                    print $"Updating ($pkg)..."
                    npm install $"($pkg)@latest"
                }
                cd ..
            },
            "rust" => {
                cd rust-env
                print "Updating Cargo dependencies..."
                cargo update
                cd ..
            },
            "go" => {
                cd go-env
                print "Updating Go modules..."
                go get -u ./...
                go mod tidy
                cd ..
            }
        }
        
        print $"✅ Updated ($env) dependencies"
    }
    
    print "\n🔍 Running post-update scan..."
    deps scan-all
}

# Clean old dependency data
def "deps cleanup" [--days: int = 90] {
    deps init
    
    let cutoff_date = (date now) - ($days * 1day)
    
    # Clean scan results
    let current_scans = (
        open .dependencies/scan_results.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $cutoff_date
    )
    
    $current_scans | each { |scan| $scan | to json --raw } | str join "\n" | save .dependencies/scan_results.jsonl
    
    # Clean alerts
    if (".dependencies/alerts.jsonl" | path exists) {
        let current_alerts = (
            open .dependencies/alerts.jsonl
            | lines
            | each { |line| $line | from json }
            | where timestamp > $cutoff_date
        )
        
        $current_alerts | each { |alert| $alert | to json --raw } | str join "\n" | save .dependencies/alerts.jsonl
    }
    
    print $"🧹 Cleaned dependency data older than ($days) days"
}

# Dependency health dashboard
def "deps dashboard" [] {
    while true {
        clear
        print "📦 Dependency Health Dashboard"
        print "=============================="
        print $"Last updated: (date now | format date '%Y-%m-%d %H:%M:%S')"
        print ""
        
        deps report --days 1 --format summary
        
        print "\nPress Ctrl+C to exit dashboard"
        sleep 30sec
    }
}

# Generate dependency optimization recommendations
def "deps optimize" [] {
    deps init
    
    let latest_scans = (
        open .dependencies/scan_results.jsonl
        | lines
        | each { |line| $line | from json }
        | group-by environment
        | transpose environment scans
        | each { |row| $row.scans | sort-by timestamp | last }
    )
    
    print "🚀 Dependency Optimization Recommendations"
    print "=========================================="
    
    for scan in $latest_scans {
        if $scan.status != "success" { continue }
        
        print $"\n📦 ($scan.environment | str upcase):"
        
        # Security recommendations
        if $scan.security_issues > 0 {
            print $"  🚨 URGENT: ($scan.security_issues) security vulnerabilities"
            print $"    → Run: deps update ($scan.environment) --security-only"
        }
        
        # Outdated packages recommendations
        if $scan.outdated_percentage > 30 {
            print $"  ⚠️  ($scan.outdated_percentage)% of packages are outdated"
            print $"    → Consider updating dependencies: deps update ($scan.environment)"
        } else if $scan.outdated_percentage > 10 {
            print $"  💡 ($scan.outdated_percentage)% of packages are outdated (acceptable)"
        }
        
        # License compliance (if available)
        if ($scan.licenses? | default [] | length) > 0 {
            let unknown_licenses = ($scan.licenses | where license == "Unknown" | length)
            if $unknown_licenses > 0 {
                print $"  📄 ($unknown_licenses) packages have unknown licenses"
                print $"    → Review license compliance for production use"
            }
        }
        
        # Package count recommendations
        if $scan.total_packages > 500 {
            print $"  📊 Large dependency tree (($scan.total_packages) packages)"
            print $"    → Consider dependency reduction to improve build times"
        }
    }
    
    # Overall recommendations
    print $"\n🎯 General Recommendations:"
    print $"  • Schedule regular dependency updates (weekly/monthly)"
    print $"  • Set up automated security scanning in CI/CD"
    print $"  • Monitor for new vulnerabilities daily"
    print $"  • Review and update dependencies before major releases"
}

# Main command dispatcher
def main [command: string, ...args] {
    match $command {
        "init" => { deps init },
        "scan-all" => { deps scan-all },
        "scan-python" => { deps scan-python },
        "scan-typescript" => { deps scan-typescript },
        "scan-rust" => { deps scan-rust },
        "scan-go" => { deps scan-go },
        "report" => { deps report ...$args },
        "update" => { deps update ...$args },
        "cleanup" => { deps cleanup ...$args },
        "dashboard" => { deps dashboard },
        "optimize" => { deps optimize },
        _ => {
            print "Comprehensive Dependency Monitoring System"
            print "Usage:"
            print "  deps init                     - Initialize dependency monitoring"
            print "  deps scan-all                 - Scan all environments for dependency issues"
            print "  deps scan-<env>               - Scan specific environment (python, typescript, rust, go)"
            print "  deps report [--days N]        - Generate dependency health report"
            print "  deps update [env] [--auto]    - Update dependencies interactively"
            print "  deps cleanup [--days N]       - Clean old dependency data"
            print "  deps dashboard                - Real-time dependency dashboard"
            print "  deps optimize                 - Get dependency optimization recommendations"
        }
    }
}