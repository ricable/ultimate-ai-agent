#!/usr/bin/env nu
# GitHub Integration Enhancement for Claude Code Hooks
# Automated issue creation, pull request management, and development workflow integration

# Initialize GitHub integration
def "github init" [] {
    mkdir -p .github-integration
    
    # Check if gh CLI is available
    if not (which gh | length > 0) {
        print "❌ GitHub CLI (gh) not found. Install with:"
        print "  brew install gh  # macOS"
        print "  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg"
        return
    }
    
    # Check if authenticated
    let auth_status = (try { gh auth status | complete } catch { {exit_code: 1} })
    if $auth_status.exit_code != 0 {
        print "🔐 GitHub CLI not authenticated. Run:"
        print "  gh auth login"
        return
    }
    
    # Initialize configuration
    if not (".github-integration/config.json" | path exists) {
        {
            version: "1.0",
            integration_config: {
                auto_create_issues: true,
                issue_labels: ["bug", "automation", "enhancement"],
                critical_severity_label: "critical",
                failure_threshold_for_issue: 3,
                duplicate_issue_window_hours: 24
            },
            issue_templates: {
                build_failure: {
                    title: "🚨 Build Failure: {environment} - {pattern}",
                    body: "## Build Failure Report\n\n**Environment:** {environment}\n**Command:** {command}\n**Pattern:** {pattern}\n**Severity:** {severity}\n\n### Error Details\n```\n{error_output}\n```\n\n### Suggested Solutions\n{solutions}\n\n### Context\n- **Timestamp:** {timestamp}\n- **Exit Code:** {exit_code}\n- **Occurrences:** {occurrences}\n\n### Automation\n- [ ] Investigate root cause\n- [ ] Implement fix\n- [ ] Add prevention pattern\n- [ ] Update documentation\n\n*This issue was automatically created by Claude Code hooks.*",
                    labels: ["bug", "build-failure", "automation"]
                },
                performance_regression: {
                    title: "⚡ Performance Regression: {environment} - {metric} degraded by {percentage}%",
                    body: "## Performance Regression Alert\n\n**Environment:** {environment}\n**Metric:** {metric}\n**Degradation:** {percentage}% slower than baseline\n**Threshold:** {threshold}\n\n### Performance Data\n- **Current:** {current_value}\n- **Baseline:** {baseline_value}\n- **Trend:** {trend}\n\n### Analysis\n{analysis}\n\n### Recommended Actions\n{recommendations}\n\n### Historical Context\n{history}\n\n### Automation\n- [ ] Profile performance bottleneck\n- [ ] Optimize critical path\n- [ ] Update performance baselines\n- [ ] Monitor improvements\n\n*This issue was automatically created by performance monitoring.*",
                    labels: ["performance", "regression", "automation"]
                },
                security_alert: {
                    title: "🛡️ Security Alert: {severity} - {pattern} detected",
                    body: "## Security Alert\n\n**Severity:** {severity}\n**Pattern:** {pattern}\n**Environment:** {environment}\n**File:** {file_path}\n\n### Security Issue Details\n```\n{security_details}\n```\n\n### Risk Assessment\n{risk_assessment}\n\n### Remediation Steps\n{remediation_steps}\n\n### Prevention\n{prevention_measures}\n\n### Automation\n- [ ] Fix security issue immediately\n- [ ] Review similar patterns\n- [ ] Update security scanning rules\n- [ ] Validate fix effectiveness\n\n⚠️ **This is a security issue - prioritize resolution**\n\n*This issue was automatically created by security scanning hooks.*",
                    labels: ["security", "vulnerability", "automation"]
                },
                dependency_alert: {
                    title: "📦 Dependency Alert: {count} issues in {environment}",
                    body: "## Dependency Health Alert\n\n**Environment:** {environment}\n**Issues Found:** {count}\n**Severity Breakdown:** {severity_breakdown}\n\n### Critical Issues\n{critical_issues}\n\n### High Priority Issues\n{high_issues}\n\n### Security Vulnerabilities\n{security_vulns}\n\n### Outdated Dependencies\n{outdated_deps}\n\n### Recommended Actions\n{recommendations}\n\n### Automation\n- [ ] Update critical dependencies\n- [ ] Test compatibility\n- [ ] Review security fixes\n- [ ] Update lock files\n\n*This issue was automatically created by dependency monitoring.*",
                    labels: ["dependencies", "maintenance", "automation"]
                }
            },
            webhook_config: {
                enabled: false,
                endpoint: "",
                secret: ""
            }
        } | save .github-integration/config.json
    }
    
    # Create issue tracking database
    if not (".github-integration/issues.jsonl" | path exists) {
        [] | save .github-integration/issues.jsonl
    }
    
    print "✅ GitHub integration initialized"
}

# Create an issue automatically based on failure patterns
def "github create-issue" [
    issue_type: string,
    environment: string,
    --data: record = {},
    --dry-run: bool = false
] {
    github init
    
    let config = (open .github-integration/config.json)
    
    if not $config.integration_config.auto_create_issues {
        print "Issue auto-creation is disabled"
        return
    }
    
    # Check for duplicate issues
    let existing_issue = (github check-duplicate $issue_type $environment $data)
    if $existing_issue != null {
        print $"Duplicate issue found: #($existing_issue.number) - ($existing_issue.title)"
        return
    }
    
    # Get issue template
    if not ($issue_type in $config.issue_templates) {
        print $"Unknown issue type: ($issue_type)"
        return
    }
    
    let template = ($config.issue_templates | get $issue_type)
    
    # Substitute template variables
    let issue_title = (github substitute-template $template.title $data)
    let issue_body = (github substitute-template $template.body $data)
    
    if $dry_run {
        print "🧪 Dry run - Would create issue:"
        print $"Title: ($issue_title)"
        print $"Labels: ($template.labels | str join ', ')"
        print $"Body:\n($issue_body)"
        return
    }
    
    # Create the issue
    let labels_arg = ($template.labels | str join ",")
    let gh_result = (try {
        gh issue create --title $issue_title --body $issue_body --label $labels_arg | complete
    } catch {
        {exit_code: 1, stderr: $in}
    })
    
    if $gh_result.exit_code == 0 {
        let issue_url = ($gh_result.stdout | str trim)
        let issue_number = ($issue_url | split row "/" | last)
        
        # Record the created issue
        let issue_record = {
            timestamp: (date now),
            issue_number: ($issue_number | into int),
            issue_url: $issue_url,
            issue_type: $issue_type,
            environment: $environment,
            title: $issue_title,
            data: $data,
            auto_created: true
        }
        
        $issue_record | to json --raw | save --append .github-integration/issues.jsonl
        
        print $"✅ Created issue #($issue_number): ($issue_title)"
        print $"   URL: ($issue_url)"
        
        # Add severity labels if critical
        if "severity" in $data and $data.severity == "critical" {
            gh issue edit $issue_number --add-label $config.integration_config.critical_severity_label
        }
        
        return $issue_record
    } else {
        print $"❌ Failed to create issue: ($gh_result.stderr)"
        return null
    }
}

# Check for duplicate issues within time window
def "github check-duplicate" [
    issue_type: string,
    environment: string,
    data: record
] {
    let config = (open .github-integration/config.json)
    let window_hours = $config.integration_config.duplicate_issue_window_hours
    let cutoff_time = (date now) - ($window_hours * 1hr)
    
    let recent_issues = (
        open .github-integration/issues.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $cutoff_time
        | where issue_type == $issue_type
        | where environment == $environment
    )
    
    # Check for similar issues based on pattern or error signature
    for issue in $recent_issues {
        if "pattern" in $data and "pattern" in $issue.data {
            if $data.pattern == $issue.data.pattern {
                return $issue
            }
        }
        
        if "error_signature" in $data and "error_signature" in $issue.data {
            if $data.error_signature == $issue.data.error_signature {
                return $issue
            }
        }
    }
    
    null
}

# Substitute template variables with actual data
def "github substitute-template" [template: string, data: record] {
    mut result = $template
    
    for key in ($data | transpose key value) {
        let placeholder = $"{($key.key)}"
        let value = if ($key.value | describe) == "string" {
            $key.value
        } else {
            $key.value | to json
        }
        
        $result = ($result | str replace --all $placeholder $value)
    }
    
    # Handle common formatting
    $result = ($result | str replace --all "\n" "\n")
    $result
}

# Monitor hook events and create issues automatically
def "github monitor-hooks" [
    --config-file: string = ".claude/settings.json"
] {
    if not ($config_file | path exists) {
        print "Claude hooks config not found"
        return
    }
    
    print "🔍 Monitoring hook events for automatic issue creation..."
    
    # This would be called by hooks or as a background process
    # For now, we'll simulate monitoring by checking recent failures
    
    # Check for critical build failures
    let recent_failures = (
        if (".failures/failure_logs.jsonl" | path exists) {
            let cutoff_time = (date now) - 1hr
            open .failures/failure_logs.jsonl
            | lines
            | each { |line| $line | from json }
            | where timestamp > $cutoff_time
            | where { |failure|
                $failure.patterns_found 
                | any { |pattern| $pattern.severity in ["critical", "high"] }
            }
        } else {
            []
        }
    )
    
    for failure in $recent_failures {
        let critical_pattern = ($failure.patterns_found | where severity in ["critical", "high"] | first)
        
        github create-issue "build_failure" $failure.environment --data {
            pattern: $critical_pattern.pattern,
            severity: $critical_pattern.severity,
            command: $failure.command,
            error_output: ($failure.output | lines | first 20 | str join "\n"),
            solutions: ($failure.suggested_solutions | str join "\n"),
            timestamp: $failure.timestamp,
            exit_code: $failure.exit_code,
            occurrences: 1
        }
    }
    
    # Check for performance regressions
    let perf_alerts = (github check-performance-regressions)
    for alert in $perf_alerts {
        github create-issue "performance_regression" $alert.environment --data $alert
    }
    
    # Check for security alerts
    let security_alerts = (github check-security-alerts)
    for alert in $security_alerts {
        github create-issue "security_alert" $alert.environment --data $alert
    }
    
    # Check for dependency issues
    let dep_alerts = (github check-dependency-alerts)
    for alert in $dep_alerts {
        github create-issue "dependency_alert" $alert.environment --data $alert
    }
}

# Check for performance regressions to create alerts
def "github check-performance-regressions" [] {
    if not (".performance/performance_logs.jsonl" | path exists) {
        return []
    }
    
    let recent_data = (
        open .performance/performance_logs.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > ((date now) - 1hr)
    )
    
    # Look for significant performance degradations
    $recent_data | where degradation_percentage > 50 | each { |entry|
        {
            environment: $entry.environment,
            metric: $entry.metric,
            percentage: $entry.degradation_percentage,
            threshold: 50,
            current_value: $entry.current_duration,
            baseline_value: $entry.baseline_duration,
            trend: "degrading",
            analysis: $"Performance degradation detected: ($entry.metric) is ($entry.degradation_percentage)% slower",
            recommendations: $entry.recommendations,
            history: $entry.history
        }
    }
}

# Check for security alerts from recent scans
def "github check-security-alerts" [] {
    if not (".security/security_logs.jsonl" | path exists) {
        return []
    }
    
    let recent_alerts = (
        open .security/security_logs.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > ((date now) - 1hr)
        | where severity in ["critical", "high"]
    )
    
    $recent_alerts | each { |alert|
        {
            environment: $alert.environment,
            severity: $alert.severity,
            pattern: $alert.pattern,
            file_path: $alert.file_path,
            security_details: $alert.details,
            risk_assessment: $alert.risk_assessment,
            remediation_steps: ($alert.suggested_fixes | str join "\n"),
            prevention_measures: $alert.prevention_measures
        }
    }
}

# Check for dependency alerts
def "github check-dependency-alerts" [] {
    if not (".dependencies/dependency_logs.jsonl" | path exists) {
        return []
    }
    
    let recent_scans = (
        open .dependencies/dependency_logs.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > ((date now) - 6hr)
    )
    
    # Group by environment and count issues
    $recent_scans | group-by environment | transpose environment scans | each { |env_group|
        let latest_scan = ($env_group.scans | sort-by timestamp | last)
        let total_issues = ($latest_scan.vulnerabilities | length) + ($latest_scan.outdated | length)
        
        if $total_issues > 5 {  # Threshold for creating issue
            {
                environment: $env_group.environment,
                count: $total_issues,
                severity_breakdown: $latest_scan.severity_summary,
                critical_issues: ($latest_scan.vulnerabilities | where severity == "critical"),
                high_issues: ($latest_scan.vulnerabilities | where severity == "high"),
                security_vulns: ($latest_scan.vulnerabilities | length),
                outdated_deps: ($latest_scan.outdated | length),
                recommendations: $latest_scan.recommendations
            }
        }
    } | where ($it != null)
}

# Create pull request for automated fixes
def "github create-fix-pr" [
    branch_name: string,
    title: string,
    description: string,
    --base: string = "main"
] {
    # Create and switch to new branch
    git checkout -b $branch_name
    
    # Commit current changes (assuming fixes have been applied)
    git add .
    git commit -m $"🤖 ($title)\n\n($description)\n\n🤖 Generated with Claude Code hooks automation"
    
    # Push branch
    git push origin $branch_name
    
    # Create pull request
    let pr_result = (try {
        gh pr create --title $title --body $description --base $base | complete
    } catch {
        {exit_code: 1, stderr: $in}
    })
    
    if $pr_result.exit_code == 0 {
        let pr_url = ($pr_result.stdout | str trim)
        print $"✅ Created pull request: ($pr_url)"
        
        # Switch back to main branch
        git checkout $base
        
        return $pr_url
    } else {
        print $"❌ Failed to create PR: ($pr_result.stderr)"
        git checkout $base
        return null
    }
}

# Update issue with new information
def "github update-issue" [
    issue_number: int,
    comment: string,
    --close: bool = false,
    --add-labels: list<string> = []
] {
    # Add comment
    gh issue comment $issue_number --body $comment
    
    # Add labels if provided
    if ($add_labels | length) > 0 {
        let labels_arg = ($add_labels | str join ",")
        gh issue edit $issue_number --add-label $labels_arg
    }
    
    # Close if requested
    if $close {
        gh issue close $issue_number --comment "Issue resolved by automation"
    }
}

# Generate GitHub integration report
def "github report" [--days: int = 7] {
    let start_date = (date now) - ($days * 1day)
    
    if not (".github-integration/issues.jsonl" | path exists) {
        print "No GitHub integration data found"
        return
    }
    
    let issues = (
        open .github-integration/issues.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $start_date
    )
    
    print $"📊 GitHub Integration Report - Last ($days) days"
    print "=" * 50
    
    if ($issues | length) == 0 {
        print "No issues created in the specified period"
        return
    }
    
    print $"Total issues created: ($issues | length)"
    
    # Issues by type
    let type_summary = (
        $issues
        | group-by issue_type
        | transpose type issues
        | each { |row|
            {
                type: $row.type,
                count: ($row.issues | length)
            }
        }
        | sort-by count --reverse
    )
    
    print "\nIssues by Type:"
    $type_summary | table
    
    # Issues by environment
    let env_summary = (
        $issues
        | group-by environment
        | transpose environment issues
        | each { |row|
            {
                environment: $row.environment,
                count: ($row.issues | length)
            }
        }
        | sort-by count --reverse
    )
    
    print "\nIssues by Environment:"
    $env_summary | table
    
    # Recent issues
    print "\nRecent Issues:"
    $issues | sort-by timestamp --reverse | first 5 | each { |issue|
        print $"  • #($issue.issue_number): ($issue.title)"
        print $"    Environment: ($issue.environment), Type: ($issue.issue_type)"
        print $"    Created: (($issue.timestamp | format date '%m-%d %H:%M'))"
        print ""
    }
}

# Test GitHub integration
def "github test" [--dry-run: bool = true] {
    print "🧪 Testing GitHub integration..."
    
    # Test authentication
    let auth_test = (try { gh auth status | complete } catch { {exit_code: 1} })
    if $auth_test.exit_code != 0 {
        print "❌ GitHub authentication failed"
        return
    }
    
    print "✅ GitHub authentication OK"
    
    # Test issue creation with sample data
    let test_data = {
        pattern: "Test pattern for integration",
        severity: "low",
        command: "test command",
        error_output: "Sample error output for testing",
        solutions: "Test solution 1\nTest solution 2",
        timestamp: (date now),
        exit_code: 1,
        occurrences: 1
    }
    
    github create-issue "build_failure" "test-env" --data $test_data --dry-run $dry_run
    
    if not $dry_run {
        print "⚠️  Created real test issue - remember to close it"
    }
}

# Main command dispatcher
def main [command: string, ...args] {
    match $command {
        "init" => { github init },
        "create-issue" => {
            if ($args | length) >= 2 {
                github create-issue $args.0 $args.1 ...(($args | skip 2))
            } else {
                print "Usage: github create-issue <issue_type> <environment> [--data record] [--dry-run]"
            }
        },
        "monitor-hooks" => { github monitor-hooks ...$args },
        "create-fix-pr" => {
            if ($args | length) >= 3 {
                github create-fix-pr $args.0 $args.1 $args.2 ...(($args | skip 3))
            } else {
                print "Usage: github create-fix-pr <branch_name> <title> <description> [--base branch]"
            }
        },
        "update-issue" => {
            if ($args | length) >= 2 {
                github update-issue ($args.0 | into int) $args.1 ...(($args | skip 2))
            } else {
                print "Usage: github update-issue <issue_number> <comment> [--close] [--add-labels list]"
            }
        },
        "report" => { github report ...$args },
        "test" => { github test ...$args },
        _ => {
            print "GitHub Integration Enhancement for Claude Code Hooks"
            print "Usage:"
            print "  github init                           - Initialize GitHub integration"
            print "  github create-issue <type> <env>     - Create issue automatically"
            print "  github monitor-hooks                 - Monitor hooks for issue creation"
            print "  github create-fix-pr <branch> <title> <desc> - Create automated fix PR"
            print "  github update-issue <num> <comment>  - Update existing issue"
            print "  github report [--days N]             - Generate integration report"
            print "  github test [--dry-run]              - Test integration functionality"
            print ""
            print "Issue Types: build_failure, performance_regression, security_alert, dependency_alert"
        }
    }
}