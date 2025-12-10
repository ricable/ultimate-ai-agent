#!/usr/bin/env nu
# Advanced Security Pattern Detection System for Polyglot Development Environment
# Detects security anti-patterns, vulnerabilities, and unsafe coding practices

# Initialize security scanning
def "security init" [] {
    mkdir .security
    
    if not (".security" | path exists) {
        mkdir .security
    }
    
    if not (".security/scan_results.jsonl" | path exists) {
        [] | save .security/scan_results.jsonl
    }
    
    if not (".security/config.json" | path exists) {
        {
            version: "1.0",
            scan_patterns: {
                python: {
                    critical: [
                        "eval\\s*\\(",
                        "exec\\s*\\(",
                        "pickle\\.loads",
                        "yaml\\.load\\s*\\(",
                        "subprocess\\.shell\\s*=\\s*True",
                        "os\\.system\\s*\\(",
                        "__import__\\s*\\("
                    ],
                    high: [
                        "input\\s*\\(",
                        "raw_input\\s*\\(",
                        "assert\\s+",
                        "random\\.random\\s*\\(",
                        "tempfile\\.mktemp",
                        "md5\\s*\\(",
                        "sha1\\s*\\("
                    ],
                    medium: [
                        "print\\s*\\(.*password",
                        "print\\s*\\(.*secret",
                        "logging\\.(info|debug).*password",
                        "except:\\s*pass",
                        "# TODO.*security",
                        "# FIXME.*security"
                    ]
                },
                typescript: {
                    critical: [
                        "eval\\s*\\(",
                        "Function\\s*\\(",
                        "innerHTML\\s*=",
                        "outerHTML\\s*=",
                        "document\\.write\\s*\\(",
                        "setTimeout\\s*\\(.*string",
                        "setInterval\\s*\\(.*string"
                    ],
                    high: [
                        "dangerouslySetInnerHTML",
                        "target\\s*=\\s*['\"]_blank['\"](?![^>]*rel\\s*=\\s*['\"][^'\"]*noopener)",
                        "Math\\.random\\s*\\(",
                        "localStorage\\.setItem.*password",
                        "sessionStorage\\.setItem.*password",
                        "console\\.log\\s*\\(.*password"
                    ],
                    medium: [
                        "alert\\s*\\(",
                        "confirm\\s*\\(",
                        "// TODO.*security",
                        "// FIXME.*security",
                        "any\\s+as\\s+",
                        "as\\s+any"
                    ]
                },
                rust: {
                    critical: [
                        "unsafe\\s*\\{",
                        "std::process::Command::new.*shell",
                        "std::ptr::",
                        "transmute",
                        "from_raw_parts"
                    ],
                    high: [
                        "unwrap\\s*\\(",
                        "expect\\s*\\(",
                        "panic!",
                        "unimplemented!",
                        "todo!",
                        "std::env::var.*unwrap"
                    ],
                    medium: [
                        "clone\\s*\\(",
                        "// TODO.*security",
                        "// FIXME.*security",
                        "println!.*password",
                        "println!.*secret"
                    ]
                },
                go: {
                    critical: [
                        "os/exec.*Shell",
                        "unsafe\\.",
                        "reflect\\.",
                        "crypto/md5",
                        "crypto/sha1",
                        "math/rand\\..*Int"
                    ],
                    high: [
                        "fmt\\.Print.*password",
                        "log\\.Print.*password",
                        "os\\.Getenv.*password",
                        "if err != nil \\{\\s*return",
                        "// TODO.*security",
                        "// FIXME.*security"
                    ],
                    medium: [
                        "panic\\s*\\(",
                        "recover\\s*\\(",
                        "defer.*recover",
                        "interface\\{\\}"
                    ]
                }
            },
            secret_patterns: [
                "(?i)(password|passwd|pwd)\\s*[:=]\\s*['\"][^'\"]{8,}['\"]",
                "(?i)(secret|token|key|api[_-]?key)\\s*[:=]\\s*['\"][^'\"]{16,}['\"]",
                "(?i)(aws[_-]?access[_-]?key[_-]?id)\\s*[:=]\\s*['\"][A-Z0-9]{20}['\"]",
                "(?i)(aws[_-]?secret[_-]?access[_-]?key)\\s*[:=]\\s*['\"][A-Za-z0-9/+=]{40}['\"]",
                "(?i)github[_-]?token\\s*[:=]\\s*['\"]ghp_[A-Za-z0-9]{36}['\"]",
                "(?i)ssh[_-]?rsa[_-]?private[_-]?key",
                "-----BEGIN (RSA )?PRIVATE KEY-----",
                "(?i)(database[_-]?url|connection[_-]?string)\\s*[:=]\\s*['\"][^'\"]*password[^'\"]*['\"]"
            ],
            file_patterns: {
                critical: [
                    "\\.env$",
                    "\\.env\\.local$",
                    "\\.env\\.production$",
                    "secrets\\..*",
                    "private[_-]?key",
                    "\\.pem$",
                    "\\.key$"
                ],
                config: [
                    "config\\.json$",
                    "config\\.yaml$",
                    "config\\.yml$",
                    "settings\\.json$",
                    "\\.config$"
                ]
            },
            severity_weights: {
                critical: 10,
                high: 5,
                medium: 2,
                low: 1
            },
            max_file_size_kb: 1024
        } | save .security/config.json
    }
    
    if not (".security/whitelist.json" | path exists) {
        {
            files: [],
            patterns: [],
            comments: "Files and patterns to exclude from security scanning"
        } | save .security/whitelist.json
    }
}

# Scan a single file for security issues
def "security scan-file" [file_path: string] {
    let config = (open .security/config.json)
    let whitelist = (open .security/whitelist.json)
    
    # Check if file is whitelisted
    if ($file_path in $whitelist.files) {
        return []
    }
    
    # Check file size
    let file_size = (ls $file_path | get size | first)
    if $file_size > ($config.max_file_size_kb * 1KB) {
        return [{
            file: $file_path,
            line: 0,
            severity: "info",
            type: "file_too_large",
            message: $"File too large for security scan (($file_size | into string))",
            pattern: "",
            recommendation: "Consider excluding large files or increasing size limit"
        }]
    }
    
    let file_content = (try { open $file_path --raw } catch { "" })
    if ($file_content | str length) == 0 {
        return []
    }
    
    let file_extension = ($file_path | path parse | get extension)
    let language = match $file_extension {
        "py" => "python",
        "js" | "ts" | "jsx" | "tsx" => "typescript",
        "rs" => "rust", 
        "go" => "go",
        _ => null
    }
    
    mut findings = []
    
    # Scan for language-specific security patterns
    if $language != null and ($language in $config.scan_patterns) {
        let patterns = ($config.scan_patterns | get $language)
        
        for severity in ["critical", "high", "medium"] {
            if $severity in $patterns {
                let severity_patterns = ($patterns | get $severity)
                
                for pattern in $severity_patterns {
                    if ($pattern in $whitelist.patterns) { continue }
                    
                    let matches = ($file_content | lines | enumerate | each { |line|
                        if ($line.item =~ $pattern) {
                            {
                                file: $file_path,
                                line: ($line.index + 1),
                                severity: $severity,
                                type: "code_pattern",
                                message: $"Potential security issue: ($pattern)",
                                pattern: $pattern,
                                content: ($line.item | str trim),
                                recommendation: (security get-recommendation $pattern $language)
                            }
                        }
                    } | compact)
                    
                    $findings = ($findings | append $matches)
                }
            }
        }
    }
    
    # Scan for secrets regardless of language
    for secret_pattern in $config.secret_patterns {
        if ($secret_pattern in $whitelist.patterns) { continue }
        
        let matches = ($file_content | lines | enumerate | each { |line|
            if ($line.item =~ $secret_pattern) {
                {
                    file: $file_path,
                    line: ($line.index + 1),
                    severity: "critical",
                    type: "secret",
                    message: "Potential secret or credential detected",
                    pattern: $secret_pattern,
                    content: ($line.item | str replace $secret_pattern "***REDACTED***"),
                    recommendation: "Remove hardcoded secrets, use environment variables or secret management"
                }
            }
        } | compact)
        
        $findings = ($findings | append $matches)
    }
    
    # Check file name patterns
    for pattern_type in ["critical", "config"] {
        let file_patterns = ($config.file_patterns | get $pattern_type)
        
        for pattern in $file_patterns {
            if ($file_path =~ $pattern) {
                let severity = if $pattern_type == "critical" { "high" } else { "medium" }
                $findings = ($findings | append [{
                    file: $file_path,
                    line: 0,
                    severity: $severity,
                    type: "sensitive_file",
                    message: $"Sensitive file type detected: ($pattern_type)",
                    pattern: $pattern,
                    content: "",
                    recommendation: "Ensure sensitive files are not committed to version control"
                }])
            }
        }
    }
    
    $findings | flatten
}

# Get security recommendation for a pattern
def "security get-recommendation" [pattern: string, language: string] {
    let recommendations = {
        "eval\\s*\\(": "Avoid eval(). Use safe alternatives like ast.literal_eval() or json.loads()",
        "exec\\s*\\(": "Avoid exec(). Consider safer alternatives or validate input thoroughly",
        "pickle\\.loads": "Avoid pickle.loads() with untrusted data. Use JSON or other safe serialization",
        "innerHTML\\s*=": "Avoid innerHTML. Use textContent or safe DOM manipulation methods",
        "setTimeout\\s*\\(.*string": "Avoid string-based setTimeout. Use function references instead",
        "unsafe\\s*\\{": "Minimize unsafe blocks. Document safety invariants and consider safe alternatives",
        "unwrap\\s*\\(": "Avoid unwrap(). Use proper error handling with match or if let",
        "Math\\.random\\s*\\(": "Don't use Math.random() for security. Use crypto.getRandomValues()",
        "os\\.system\\s*\\(": "Avoid os.system(). Use subprocess with shell=False instead",
        "fmt\\.Print.*password": "Don't log passwords. Use structured logging and sanitize sensitive data"
    }
    
    $recommendations | get -i $pattern | default "Review this pattern for security implications"
}

# Scan directory recursively
def "security scan-directory" [
    directory: string,
    --exclude: list<string> = [],
    --include-extensions: list<string> = ["py", "js", "ts", "jsx", "tsx", "rs", "go", "env", "json", "yaml", "yml"]
] {
    security init
    
    let files = (
        glob ($directory + "/**/*")
        | where { |file| ($file | path type) == "file" }
        | where { |file| 
            let ext = ($file | path parse | get extension)
            $ext in $include_extensions
        }
        | where { |file|
            let excluded = ($exclude | any { |pattern| $file | str contains $pattern })
            not $excluded
        }
    )
    
    print $"🔍 Scanning ($files | length) files for security issues..."
    
    mut all_findings = []
    let total_files = ($files | length)
    
    for file in ($files | enumerate) {
        if ($file.index mod 10) == 0 {
            print $"  Progress: ($file.index)/($total_files) files"
        }
        
        let findings = (security scan-file $file.item)
        $all_findings = ($all_findings | append $findings)
    }
    
    $all_findings | flatten
}

# Scan all environments
def "security scan-all" [] {
    security init
    
    print "🛡️  Running comprehensive security scan across all environments..."
    
    let environments = ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env"]
    mut all_results = []
    
    for env in $environments {
        if not ($env | path exists) {
            print $"⚠️  Skipping ($env) - directory not found"
            continue
        }
        
        print $"🔍 Scanning ($env)..."
        
        let exclude_patterns = [
            "node_modules",
            ".devbox",
            ".git", 
            "__pycache__",
            "target",
            ".venv",
            ".uv-cache",
            "dist",
            "build"
        ]
        
        let findings = (security scan-directory $env --exclude $exclude_patterns)
        
        let result = {
            environment: $env,
            timestamp: (date now),
            total_files_scanned: (glob ($env + "/**/*") | where { |f| ($f | path type) == "file" } | length),
            findings: $findings,
            summary: (security summarize-findings $findings)
        }
        
        $all_results = ($all_results | append $result)
        
        # Show immediate critical issues
        let critical_findings = ($findings | where severity == "critical")
        if ($critical_findings | length) > 0 {
            print $"🚨 CRITICAL: ($critical_findings | length) critical security issues found in ($env)"
        }
    }
    
    # Save results
    $all_results | each { |result| 
        $result | to json --raw | save --append .security/scan_results.jsonl
    }
    
    # Generate summary
    let total_findings = ($all_results | get findings | flatten | length)
    let critical_count = ($all_results | get findings | flatten | where severity == "critical" | length)
    
    print $"✅ Security scan completed"
    print $"   Total findings: ($total_findings)"
    print $"   Critical issues: ($critical_count)"
    
    if $critical_count > 0 {
        print "🚨 URGENT: Critical security issues require immediate attention!"
    }
    
    $all_results
}

# Summarize findings by severity
def "security summarize-findings" [findings: list] {
    let config = (open .security/config.json)
    
    let summary = ($findings | group-by severity | transpose severity items | each { |group|
        {
            severity: $group.severity,
            count: ($group.items | length),
            weight: (($group.items | length) * ($config.severity_weights | get $group.severity)),
            types: ($group.items | group-by type | transpose type count | each { |t| { type: $t.type, count: ($t.count | length) } })
        }
    })
    
    let total_weight = ($summary | get weight | math sum)
    let security_score = (100 - ([$total_weight, 100] | math min))
    
    {
        by_severity: $summary,
        total_findings: ($findings | length),
        security_score: $security_score,
        risk_level: (if $security_score > 80 { "low" } else if $security_score > 60 { "medium" } else if $security_score > 40 { "high" } else { "critical" })
    }
}

# Generate security report
def "security report" [
    --days: int = 7,
    --environment: string = "all",
    --format: string = "summary",
    --severity: string = "all"
] {
    security init
    
    let start_date = (date now) - ($days * 1day)
    
    let scan_data = (
        open .security/scan_results.jsonl
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
        print $"No security scan data found for the last ($days) days"
        return
    }
    
    match $format {
        "summary" => {
            print $"🛡️  Security Report - Last ($days) days"
            print "=" * 45
            
            # Get latest scan for each environment
            let latest_scans = (
                $filtered_data
                | group-by environment
                | transpose environment scans
                | each { |row|
                    $row.scans | sort-by timestamp | last
                }
            )
            
            print "Current Security Status:"
            $latest_scans | each { |scan|
                let summary = $scan.summary
                let risk_emoji = match $summary.risk_level {
                    "low" => "🟢",
                    "medium" => "🟡", 
                    "high" => "🟠",
                    "critical" => "🔴"
                }
                
                let critical_count = ($summary.by_severity | where severity == "critical" | get count | math sum)
                let high_count = ($summary.by_severity | where severity == "high" | get count | math sum)
                
                print $"($risk_emoji) ($scan.environment): Score ($summary.security_score)/100, Risk: ($summary.risk_level)"
                if $critical_count > 0 {
                    print $"    🚨 ($critical_count) critical issues"
                }
                if $high_count > 0 {
                    print $"    ⚠️  ($high_count) high-severity issues"
                }
            }
            
            # Top security issues across environments
            let all_findings = ($latest_scans | get findings | flatten)
            let critical_findings = ($all_findings | where severity == "critical")
            
            if ($critical_findings | length) > 0 {
                print $"\n🚨 Critical Security Issues (Top 5):"
                $critical_findings 
                | group-by pattern 
                | transpose pattern instances
                | each { |issue| 
                    {
                        pattern: $issue.pattern,
                        count: ($issue.instances | length),
                        files: ($issue.instances | get file | uniq | length)
                    }
                }
                | sort-by count --reverse
                | first 5
                | each { |issue|
                    print $"  • ($issue.pattern): ($issue.count) instances across ($issue.files) files"
                }
            }
            
            # Security trends
            if ($filtered_data | length) > 1 {
                print $"\n📈 Security Trends:"
                let trend_data = (
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
                                score_change: ($last.summary.security_score - $first.summary.security_score),
                                findings_change: ($last.summary.total_findings - $first.summary.total_findings)
                            }
                        }
                    }
                    | compact
                )
                
                $trend_data | each { |trend|
                    let score_emoji = if $trend.score_change > 0 { "📈🟢" } else if $trend.score_change < 0 { "📉🔴" } else { "➡️" }
                    print $"  ($trend.environment): Security score ($score_emoji) ($trend.score_change), Findings: ($trend.findings_change)"
                }
            }
        },
        "detailed" => {
            let all_findings = ($filtered_data | get findings | flatten)
            
            let filtered_findings = if $severity == "all" {
                $all_findings
            } else {
                $all_findings | where severity == $severity
            }
            
            $filtered_findings | group-by severity | transpose severity findings | each { |group|
                print $"\n($group.severity | str upcase) SEVERITY:"
                $group.findings | each { |finding|
                    print $"  File: ($finding.file):($finding.line)"
                    print $"  Issue: ($finding.message)"
                    print $"  Recommendation: ($finding.recommendation)"
                    print ""
                }
            }
        },
        "json" => {
            $filtered_data | to json
        },
        _ => {
            print "Invalid format. Use: summary, detailed, json"
        }
    }
}

# Fix security issues automatically where possible
def "security fix" [
    --environment: string = "all",
    --severity: string = "critical",
    --auto = false,
    --dry-run = false
] {
    print $"🔧 Security Auto-Fix (Severity: ($severity))"
    
    if $dry_run {
        print "🚦 DRY RUN MODE - No changes will be made"
    }
    
    # Get latest scan results
    let latest_scans = (
        open .security/scan_results.jsonl
        | lines
        | each { |line| $line | from json }
        | group-by environment
        | transpose environment scans
        | each { |row| $row.scans | sort-by timestamp | last }
    )
    
    let scans_to_fix = if $environment == "all" {
        $latest_scans
    } else {
        $latest_scans | where environment == $environment
    }
    
    for scan in $scans_to_fix {
        let fixable_findings = (
            $scan.findings 
            | where severity == $severity 
            | where type == "secret" or type == "code_pattern"
        )
        
        if ($fixable_findings | length) == 0 {
            print $"✅ No fixable ($severity) issues in ($scan.environment)"
            continue
        }
        
        print $"🔧 Found ($fixable_findings | length) fixable issues in ($scan.environment)"
        
        if not $auto and not $dry_run {
            let confirm = (input $"Proceed with fixing ($scan.environment)? (y/n): ")
            if $confirm != "y" and $confirm != "yes" {
                print "Skipping fixes."
                continue
            }
        }
        
        # Group findings by file for efficient processing
        let files_to_fix = ($fixable_findings | group-by file | transpose file findings)
        
        for file_group in $files_to_fix {
            let file_path = $file_group.file
            print $"  📄 Fixing ($file_path)..."
            
            if $dry_run {
                $file_group.findings | each { |finding|
                    print $"    [DRY RUN] Would fix: ($finding.message) at line ($finding.line)"
                }
                continue
            }
            
            # Apply fixes based on finding type
            for finding in $file_group.findings {
                match $finding.type {
                    "secret" => {
                        print $"    ⚠️  SECRET DETECTED: Manual review required for ($finding.message)"
                        print $"       Line ($finding.line): Consider using environment variables"
                    },
                    "code_pattern" => {
                        print $"    🔍 CODE PATTERN: ($finding.message) at line ($finding.line)"
                        print $"       Recommendation: ($finding.recommendation)"
                    }
                }
            }
        }
    }
    
    if not $dry_run {
        print "🔄 Running security scan to verify fixes..."
        security scan-all
    }
}

# Add patterns to whitelist
def "security whitelist" [
    --file: string,
    --pattern: string,
    --reason: string = "Manual review completed"
] {
    mut whitelist = (open .security/whitelist.json)
    
    if $file != null {
        if not ($file in $whitelist.files) {
            $whitelist.files = ($whitelist.files | append $file)
            print $"✅ Added file to whitelist: ($file)"
        } else {
            print $"ℹ️  File already whitelisted: ($file)"
        }
    }
    
    if $pattern != null {
        if not ($pattern in $whitelist.patterns) {
            $whitelist.patterns = ($whitelist.patterns | append $pattern)
            print $"✅ Added pattern to whitelist: ($pattern)"
        } else {
            print $"ℹ️  Pattern already whitelisted: ($pattern)"
        }
    }
    
    $whitelist | save .security/whitelist.json
}

# Clean old security scan data
def "security cleanup" [--days: int = 90] {
    security init
    
    let cutoff_date = (date now) - ($days * 1day)
    
    let current_scans = (
        open .security/scan_results.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $cutoff_date
    )
    
    $current_scans | each { |scan| $scan | to json --raw } | str join "\n" | save .security/scan_results.jsonl
    
    print $"🧹 Cleaned security data older than ($days) days"
}

# Main command
def main [file_path?: string] {
    if $file_path == null {
        print "Usage: security scan-file <file_path>"
        return
    }
    
    security init
    let findings = (security scan-file $file_path)
    
    if ($findings | length) == 0 {
        print "✅ No security issues found!"
    } else {
        print $"⚠️ Found ($findings | length) security issues:"
        $findings | each { |finding|
            print $"  Line ($finding.line): ($finding.severity) - ($finding.message)"
        }
    }
}