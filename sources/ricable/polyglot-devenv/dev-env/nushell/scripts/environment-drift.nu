#!/usr/bin/env nu
# Environment Consistency & Drift Detection System for Polyglot Development Environment
# Monitors configuration drift, ensures environment consistency, and tracks changes

# Initialize drift detection
def "drift init" [] {
    mkdir -p .environment
    
    if not (".environment/snapshots.jsonl" | path exists) {
        [] | save .environment/snapshots.jsonl
    }
    
    if not (".environment/config.json" | path exists) {
        {
            version: "1.0",
            environments: {
                python: {
                    path: "python-env",
                    config_files: ["devbox.json", "pyproject.toml", "uv.lock"],
                    tools: ["python", "uv", "ruff", "mypy"],
                    critical_paths: ["src/", "tests/"]
                },
                typescript: {
                    path: "typescript-env", 
                    config_files: ["devbox.json", "package.json", "package-lock.json", "tsconfig.json"],
                    tools: ["node", "npm", "typescript", "eslint"],
                    critical_paths: ["src/", "tests/"]
                },
                rust: {
                    path: "rust-env",
                    config_files: ["devbox.json", "Cargo.toml", "Cargo.lock"],
                    tools: ["rustc", "cargo", "clippy", "rustfmt"],
                    critical_paths: ["src/", "tests/"]
                },
                go: {
                    path: "go-env",
                    config_files: ["devbox.json", "go.mod", "go.sum"],
                    tools: ["go", "golangci-lint"],
                    critical_paths: ["cmd/", "pkg/", "internal/"]
                },
                nushell: {
                    path: "nushell-env",
                    config_files: ["devbox.json", "common.nu"],
                    tools: ["nu", "teller"],
                    critical_paths: ["scripts/", "config/"]
                }
            },
            drift_thresholds: {
                config_changes: 0,      # No config changes allowed without tracking
                tool_version_drift: 1,  # Max minor version difference
                missing_files: 0,       # No critical files should be missing
                permission_changes: 0   # No permission changes allowed
            },
            baseline_retention_days: 30,
            alert_channels: ["console", "file"]
        } | save .environment/config.json
    }
    
    if not (".environment/baseline.json" | path exists) {
        print "📷 Creating initial baseline snapshot..."
        drift snapshot --save-as-baseline
    }
}

# Create environment snapshot
def "drift snapshot" [
    --environment: string = "all",
    --save-as-baseline: bool = false
] {
    drift init
    
    let config = (open .environment/config.json)
    let environments_to_scan = if $environment == "all" {
        ($config.environments | transpose name details | get name)
    } else {
        [$environment]
    }
    
    mut snapshots = []
    
    for env_name in $environments_to_scan {
        if not ($env_name in $config.environments) {
            print $"⚠️  Unknown environment: ($env_name)"
            continue
        }
        
        let env_config = ($config.environments | get $env_name)
        let env_path = $env_config.path
        
        if not ($env_path | path exists) {
            print $"⚠️  Environment path not found: ($env_path)"
            continue
        }
        
        print $"📷 Creating snapshot for ($env_name)..."
        
        # Collect configuration file hashes
        let config_hashes = ($env_config.config_files | each { |file|
            let file_path = ($env_path | path join $file)
            if ($file_path | path exists) {
                {
                    file: $file,
                    path: $file_path,
                    hash: (open $file_path --raw | hash sha256),
                    size: (ls $file_path | get size | first),
                    modified: (ls $file_path | get modified | first)
                }
            } else {
                {
                    file: $file,
                    path: $file_path,
                    hash: "MISSING",
                    size: 0,
                    modified: null
                }
            }
        })
        
        # Check tool versions
        let tool_versions = ($env_config.tools | each { |tool|
            let version = (try {
                match $tool {
                    "python" => (python --version | str replace "Python " ""),
                    "uv" => (uv --version | str replace "uv " ""),
                    "node" => (node --version | str replace "v" ""),
                    "npm" => (npm --version),
                    "typescript" => (tsc --version | str replace "Version " ""),
                    "rustc" => (rustc --version | split row " " | get 1),
                    "cargo" => (cargo --version | split row " " | get 1),
                    "go" => (go version | split row " " | get 2 | str replace "go" ""),
                    "nu" => (nu --version | lines | first | split row " " | get 1),
                    _ => (nu -c $"($tool) --version" | lines | first)
                }
            } catch {
                "NOT_INSTALLED"
            })
            
            {
                tool: $tool,
                version: $version,
                available: ($version != "NOT_INSTALLED")
            }
        })
        
        # Scan critical paths
        let path_info = ($env_config.critical_paths | each { |path|
            let full_path = ($env_path | path join $path)
            if ($full_path | path exists) {
                let files = (try {
                    glob ($full_path + "/**/*") 
                    | where { |f| ($f | path type) == "file" }
                    | length
                } catch { 0 })
                
                {
                    path: $path,
                    exists: true,
                    file_count: $files,
                    total_size: (try {
                        du $full_path | get physical | first
                    } catch { 0 })
                }
            } else {
                {
                    path: $path,
                    exists: false,
                    file_count: 0,
                    total_size: 0
                }
            }
        })
        
        # Check devbox environment
        let devbox_info = (try {
            cd $env_path
            let info = (devbox info --json | from json)
            cd ..
            
            {
                packages: ($info.packages? | default []),
                shell_init_hook: ($info.shell?.init_hook? | default []),
                scripts: ($info.shell?.scripts? | default {}),
                env_vars: ($info.env? | default {})
            }
        } catch {
            {
                packages: [],
                shell_init_hook: [],
                scripts: {},
                env_vars: {}
            }
        })
        
        let snapshot = {
            environment: $env_name,
            timestamp: (date now),
            config_files: $config_hashes,
            tool_versions: $tool_versions,
            critical_paths: $path_info,
            devbox_info: $devbox_info,
            system_info: {
                platform: (sys | get host.name),
                kernel: (sys | get host.kernel_version),
                arch: (sys | get host.cpu | first | get arch?),
                total_memory: (sys | get host.memory.total)
            }
        }
        
        $snapshots = ($snapshots | append $snapshot)
    }
    
    # Save snapshots
    if $save_as_baseline {
        {
            created: (date now),
            snapshots: $snapshots
        } | save .environment/baseline.json
        print "✅ Baseline snapshot saved"
    } else {
        $snapshots | each { |snapshot| 
            $snapshot | to json --raw | save --append .environment/snapshots.jsonl
        }
        print $"✅ Snapshot saved for ($snapshots | length) environments"
    }
    
    $snapshots
}

# Detect drift from baseline
def "drift detect" [
    --environment: string = "all",
    --threshold: string = "medium"  # strict, medium, relaxed
] {
    drift init
    
    if not (".environment/baseline.json" | path exists) {
        print "❌ No baseline found. Run 'drift snapshot --save-as-baseline' first"
        return []
    }
    
    let baseline = (open .environment/baseline.json)
    let current_snapshots = (drift snapshot --environment $environment)
    
    let thresholds = match $threshold {
        "strict" => { config: 0, tools: 0, paths: 0 },
        "medium" => { config: 1, tools: 2, paths: 1 },
        "relaxed" => { config: 3, tools: 5, paths: 2 }
    }
    
    mut drift_results = []
    
    for current in $current_snapshots {
        let baseline_env = ($baseline.snapshots | where environment == $current.environment | first)
        
        if $baseline_env == null {
            $drift_results = ($drift_results | append {
                environment: $current.environment,
                status: "new_environment",
                drift_score: 0,
                issues: []
            })
            continue
        }
        
        mut issues = []
        mut drift_score = 0
        
        # Check configuration file changes
        for current_file in $current.config_files {
            let baseline_file = ($baseline_env.config_files | where file == $current_file.file | first)
            
            if $baseline_file == null {
                $issues = ($issues | append {
                    type: "new_config_file",
                    severity: "medium",
                    description: $"New configuration file: ($current_file.file)",
                    recommendation: "Review new file and update baseline if intentional"
                })
                $drift_score = ($drift_score + 2)
            } else if $current_file.hash != $baseline_file.hash {
                $issues = ($issues | append {
                    type: "config_file_changed",
                    severity: "high",
                    description: $"Configuration file modified: ($current_file.file)",
                    recommendation: "Review changes and update baseline if approved"
                })
                $drift_score = ($drift_score + 5)
            }
        }
        
        # Check for missing configuration files
        for baseline_file in $baseline_env.config_files {
            let current_file = ($current.config_files | where file == $baseline_file.file | first)
            if $current_file == null or $current_file.hash == "MISSING" {
                $issues = ($issues | append {
                    type: "missing_config_file",
                    severity: "critical",
                    description: $"Missing configuration file: ($baseline_file.file)",
                    recommendation: "Restore missing file or update environment configuration"
                })
                $drift_score = ($drift_score + 10)
            }
        }
        
        # Check tool version drift
        for current_tool in $current.tool_versions {
            let baseline_tool = ($baseline_env.tool_versions | where tool == $current_tool.tool | first)
            
            if $baseline_tool != null and $current_tool.version != $baseline_tool.version {
                let severity = if $current_tool.available and $baseline_tool.available { "medium" } else { "high" }
                $issues = ($issues | append {
                    type: "tool_version_drift",
                    severity: $severity,
                    description: $"Tool version changed: ($current_tool.tool) ($baseline_tool.version) → ($current_tool.version)",
                    recommendation: "Verify compatibility and update team environments"
                })
                $drift_score = ($drift_score + 3)
            } else if not $current_tool.available and $baseline_tool.available {
                $issues = ($issues | append {
                    type: "missing_tool",
                    severity: "critical",
                    description: $"Tool no longer available: ($current_tool.tool)",
                    recommendation: "Reinstall missing tool or update environment configuration"
                })
                $drift_score = ($drift_score + 8)
            }
        }
        
        # Check critical path changes
        for current_path in $current.critical_paths {
            let baseline_path = ($baseline_env.critical_paths | where path == $current_path.path | first)
            
            if $baseline_path != null {
                let file_count_diff = ($current_path.file_count - $baseline_path.file_count)
                if ($file_count_diff | math abs) > $thresholds.paths {
                    $issues = ($issues | append {
                        type: "critical_path_changed",
                        severity: "medium",
                        description: $"File count changed in ($current_path.path): ($file_count_diff) files",
                        recommendation: "Review structural changes to critical paths"
                    })
                    $drift_score = ($drift_score + 2)
                }
            }
        }
        
        # Check devbox package changes
        let baseline_packages = ($baseline_env.devbox_info.packages | sort)
        let current_packages = ($current.devbox_info.packages | sort)
        
        if $baseline_packages != $current_packages {
            $issues = ($issues | append {
                type: "devbox_packages_changed",
                severity: "high",
                description: "Devbox packages configuration changed",
                recommendation: "Review package changes and update team environments"
            })
            $drift_score = ($drift_score + 4)
        }
        
        # Determine overall status
        let status = if $drift_score == 0 {
            "consistent"
        } else if $drift_score <= 5 {
            "minor_drift"
        } else if $drift_score <= 15 {
            "moderate_drift"
        } else {
            "major_drift"
        }
        
        $drift_results = ($drift_results | append {
            environment: $current.environment,
            status: $status,
            drift_score: $drift_score,
            issues: $issues,
            snapshot_comparison: {
                baseline_date: $baseline.created,
                current_date: $current.timestamp
            }
        })
    }
    
    $drift_results
}

# Generate drift report
def "drift report" [
    --environment: string = "all",
    --format: string = "summary",
    --days: int = 7
] {
    drift init
    
    let drift_results = (drift detect --environment $environment)
    
    match $format {
        "summary" => {
            print "🔍 Environment Drift Analysis"
            print "============================="
            
            if ($drift_results | length) == 0 {
                print "No environments to analyze"
                return
            }
            
            # Overall status
            let status_summary = (
                $drift_results 
                | group-by status 
                | transpose status count 
                | each { |row| 
                    { 
                        status: $row.status, 
                        count: ($row.count | length),
                        environments: ($row.count | get environment)
                    } 
                }
            )
            
            print "Environment Status:"
            $status_summary | each { |summary|
                let status_emoji = match $summary.status {
                    "consistent" => "🟢",
                    "minor_drift" => "🟡",
                    "moderate_drift" => "🟠",
                    "major_drift" => "🔴",
                    "new_environment" => "🆕"
                }
                print $"($status_emoji) ($summary.status): ($summary.count) environments"
                if $summary.status != "consistent" {
                    $summary.environments | each { |env| print $"   • ($env)" }
                }
            }
            
            # Critical issues that need immediate attention
            let critical_issues = (
                $drift_results 
                | get issues 
                | flatten 
                | where severity == "critical"
            )
            
            if ($critical_issues | length) > 0 {
                print $"\n🚨 Critical Issues Requiring Immediate Attention:"
                $critical_issues | each { |issue|
                    print $"  • ($issue.description)"
                    print $"    ⚡ ($issue.recommendation)"
                }
            }
            
            # Environment-specific details
            print $"\nEnvironment Details:"
            $drift_results | each { |result|
                if $result.status != "consistent" {
                    print $"\n🔧 ($result.environment) - ($result.status) (Score: ($result.drift_score))"
                    $result.issues | each { |issue|
                        let severity_emoji = match $issue.severity {
                            "critical" => "🚨",
                            "high" => "⚠️ ",
                            "medium" => "⚡",
                            "low" => "ℹ️ "
                        }
                        print $"  ($severity_emoji) ($issue.description)"
                    }
                }
            }
        },
        "detailed" => {
            $drift_results | to json
        },
        "json" => {
            $drift_results | to json
        },
        _ => {
            print "Invalid format. Use: summary, detailed, json"
        }
    }
}

# Synchronize environment from baseline or another environment
def "drift sync" [
    target_environment: string,
    --from-baseline: bool = true,
    --from-environment: string = "",
    --dry-run: bool = false,
    --auto-approve: bool = false
] {
    drift init
    
    let config = (open .environment/config.json)
    
    if not ($target_environment in $config.environments) {
        print $"❌ Unknown environment: ($target_environment)"
        return
    }
    
    let source_data = if $from_baseline {
        let baseline = (open .environment/baseline.json)
        $baseline.snapshots | where environment == $target_environment | first
    } else if $from_environment != "" {
        # Get latest snapshot from another environment
        let snapshots = (
            open .environment/snapshots.jsonl
            | lines
            | each { |line| $line | from json }
            | where environment == $from_environment
            | sort-by timestamp
        )
        $snapshots | last
    } else {
        print "❌ Must specify either --from-baseline or --from-environment"
        return
    }
    
    if $source_data == null {
        print $"❌ No source data found for synchronization"
        return
    }
    
    let target_path = ($config.environments | get $target_environment | get path)
    
    print $"🔄 Synchronizing ($target_environment) from " + (if $from_baseline { "baseline" } else { $from_environment })
    
    if $dry_run {
        print "🚦 DRY RUN MODE - No changes will be made"
    }
    
    # Check current state
    let current_snapshot = (drift snapshot --environment $target_environment) | first
    
    # Compare and sync configuration files
    for source_file in $source_data.config_files {
        let current_file = ($current_snapshot.config_files | where file == $source_file.file | first)
        
        if $current_file == null or $current_file.hash != $source_file.hash {
            print $"📄 Configuration file needs sync: ($source_file.file)"
            
            if not $dry_run {
                if not $auto_approve {
                    let confirm = (input $"Sync ($source_file.file)? (y/n): ")
                    if $confirm != "y" and $confirm != "yes" {
                        print "Skipping file."
                        continue
                    }
                }
                
                # Backup current file if it exists
                if $current_file != null and $current_file.hash != "MISSING" {
                    let backup_path = ($source_file.path + ".backup." + (date now | format date '%Y%m%d_%H%M%S'))
                    cp $source_file.path $backup_path
                    print $"  💾 Backed up to: ($backup_path)"
                }
                
                print $"  ✅ Would sync ($source_file.file) (dry-run mode prevents actual sync)"
            }
        }
    }
    
    # Check tool versions
    for source_tool in $source_data.tool_versions {
        let current_tool = ($current_snapshot.tool_versions | where tool == $source_tool.tool | first)
        
        if $current_tool == null or $current_tool.version != $source_tool.version {
            print $"🔧 Tool version mismatch: ($source_tool.tool)"
            print $"  Current: ($current_tool?.version? | default 'NOT_INSTALLED')"
            print $"  Target: ($source_tool.version)"
            print $"  💡 Recommendation: Update tool manually or run environment setup"
        }
    }
    
    # Check devbox packages
    let source_packages = ($source_data.devbox_info.packages | sort)
    let current_packages = ($current_snapshot.devbox_info.packages | sort)
    
    if $source_packages != $current_packages {
        print $"📦 Devbox packages need sync"
        print $"  🔧 Run: cd ($target_path) && devbox update"
    }
    
    if not $dry_run {
        print $"\n🔄 Running post-sync verification..."
        let post_sync_results = (drift detect --environment $target_environment)
        let env_result = ($post_sync_results | where environment == $target_environment | first)
        
        if $env_result.status == "consistent" {
            print $"✅ ($target_environment) is now consistent with source"
        } else {
            print $"⚠️  Some drift remains in ($target_environment):"
            $env_result.issues | each { |issue|
                print $"  • ($issue.description)"
            }
        }
    }
}

# Update baseline with current state
def "drift update-baseline" [
    --environment: string = "all",
    --backup: bool = true
] {
    drift init
    
    if $backup and (".environment/baseline.json" | path exists) {
        let backup_path = (".environment/baseline.backup." + (date now | format date '%Y%m%d_%H%M%S') + ".json")
        cp .environment/baseline.json $backup_path
        print $"💾 Baseline backed up to: ($backup_path)"
    }
    
    print $"📷 Creating new baseline from current state..."
    drift snapshot --environment $environment --save-as-baseline
    
    print $"✅ Baseline updated successfully"
}

# Monitor for real-time drift
def "drift monitor" [
    --interval: int = 300,  # 5 minutes
    --environment: string = "all"
] {
    print $"👀 Starting drift monitoring (interval: ($interval)s)"
    print "Press Ctrl+C to stop monitoring"
    
    while true {
        let results = (drift detect --environment $environment)
        let drifted = ($results | where status != "consistent")
        
        if ($drifted | length) > 0 {
            let timestamp = (date now | format date '%H:%M:%S')
            print $"[$timestamp] ⚠️  Drift detected in ($drifted | length) environments:"
            $drifted | each { |env|
                print $"  • ($env.environment): ($env.status) (score: ($env.drift_score))"
            }
        } else {
            let timestamp = (date now | format date '%H:%M:%S')
            print $"[$timestamp] ✅ All environments consistent"
        }
        
        sleep ($interval)sec
    }
}

# Clean old snapshots
def "drift cleanup" [--days: int = 30] {
    drift init
    
    let cutoff_date = (date now) - ($days * 1day)
    
    let current_snapshots = (
        open .environment/snapshots.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $cutoff_date
    )
    
    $current_snapshots | each { |snapshot| $snapshot | to json --raw } | str join "\n" | save .environment/snapshots.jsonl
    
    print $"🧹 Cleaned snapshots older than ($days) days"
}

# Main command dispatcher
def main [command: string, ...args] {
    match $command {
        "init" => { drift init },
        "snapshot" => { drift snapshot ...$args },
        "detect" => { drift detect ...$args },
        "report" => { drift report ...$args },
        "sync" => { 
            if ($args | length) >= 1 {
                drift sync $args.0 ...(($args | skip 1))
            } else {
                print "Usage: drift sync <target_environment> [--from-baseline|--from-environment <env>]"
            }
        },
        "update-baseline" => { drift update-baseline ...$args },
        "monitor" => { drift monitor ...$args },
        "cleanup" => { drift cleanup ...$args },
        _ => {
            print "Environment Consistency & Drift Detection System"
            print "Usage:"
            print "  drift init                        - Initialize drift detection"
            print "  drift snapshot [--save-as-baseline] - Create environment snapshot"
            print "  drift detect [--environment env]  - Detect drift from baseline"
            print "  drift report [--format summary]   - Generate drift analysis report"
            print "  drift sync <env> [--from-baseline] - Synchronize environment"
            print "  drift update-baseline [--backup] - Update baseline with current state"
            print "  drift monitor [--interval N]     - Real-time drift monitoring"
            print "  drift cleanup [--days N]         - Clean old snapshot data"
        }
    }
}