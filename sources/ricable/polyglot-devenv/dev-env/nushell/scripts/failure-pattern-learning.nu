#!/usr/bin/env nu
# Failure Pattern Learning and Analysis System for Polyglot Development Environment
# Analyzes build/test failures, identifies common patterns, and suggests fixes based on historical data

# Initialize failure pattern learning
def "failure init" [] {
    mkdir -p .failures
    
    if not (".failures/failure_logs.jsonl" | path exists) {
        [] | save .failures/failure_logs.jsonl
    }
    
    if not (".failures/pattern_db.json" | path exists) {
        {
            version: "1.0",
            patterns: {
                python: {
                    common_failures: [
                        {
                            pattern: "ModuleNotFoundError: No module named",
                            category: "dependency",
                            severity: "high",
                            solutions: [
                                "Install missing package: uv add <package>",
                                "Check if package is in requirements",
                                "Verify virtual environment activation"
                            ],
                            keywords: ["module", "import", "not found"]
                        },
                        {
                            pattern: "SyntaxError: invalid syntax",
                            category: "syntax",
                            severity: "high",
                            solutions: [
                                "Check Python version compatibility",
                                "Verify syntax for the Python version",
                                "Look for missing colons, brackets, or quotes"
                            ],
                            keywords: ["syntax", "invalid", "unexpected"]
                        },
                        {
                            pattern: "IndentationError",
                            category: "formatting",
                            severity: "medium",
                            solutions: [
                                "Fix indentation using consistent spaces/tabs",
                                "Run formatter: uv run ruff format",
                                "Use editor with Python indentation support"
                            ],
                            keywords: ["indent", "tab", "space"]
                        },
                        {
                            pattern: "TypeError.*takes.*positional argument",
                            category: "api",
                            severity: "medium",
                            solutions: [
                                "Check function signature and arguments",
                                "Verify API documentation for changes",
                                "Update function calls to match signature"
                            ],
                            keywords: ["argument", "parameter", "signature"]
                        }
                    ]
                },
                typescript: {
                    common_failures: [
                        {
                            pattern: "Cannot find module.*or its corresponding type declarations",
                            category: "dependency",
                            severity: "high",
                            solutions: [
                                "Install missing package: npm install <package>",
                                "Install type definitions: npm install @types/<package>",
                                "Check import path and spelling"
                            ],
                            keywords: ["module", "import", "types", "declaration"]
                        },
                        {
                            pattern: "Property.*does not exist on type",
                            category: "typing",
                            severity: "medium",
                            solutions: [
                                "Check object type definition",
                                "Use optional chaining: obj?.property",
                                "Add type assertion if certain: obj as Type"
                            ],
                            keywords: ["property", "type", "exist"]
                        },
                        {
                            pattern: "Argument of type.*is not assignable to parameter of type",
                            category: "typing",
                            severity: "medium",
                            solutions: [
                                "Check parameter types match",
                                "Use type conversion if needed",
                                "Update interface definitions"
                            ],
                            keywords: ["argument", "assignable", "parameter"]
                        },
                        {
                            pattern: "Expected.*arguments, but got",
                            category: "api",
                            severity: "medium",
                            solutions: [
                                "Check function signature",
                                "Verify number of arguments passed",
                                "Look for API changes in dependencies"
                            ],
                            keywords: ["arguments", "expected", "signature"]
                        }
                    ]
                },
                rust: {
                    common_failures: [
                        {
                            pattern: "cannot find.*in this scope",
                            category: "scope",
                            severity: "high",
                            solutions: [
                                "Add appropriate use statement",
                                "Check if item is public",
                                "Verify crate dependencies"
                            ],
                            keywords: ["scope", "find", "use", "import"]
                        },
                        {
                            pattern: "borrow checker",
                            category: "ownership",
                            severity: "high",
                            solutions: [
                                "Review ownership and borrowing rules",
                                "Use references appropriately",
                                "Consider using Rc/Arc for shared ownership"
                            ],
                            keywords: ["borrow", "ownership", "lifetime"]
                        },
                        {
                            pattern: "trait.*is not implemented",
                            category: "traits",
                            severity: "medium",
                            solutions: [
                                "Implement required trait",
                                "Add trait bounds to generics",
                                "Use derive macro if available"
                            ],
                            keywords: ["trait", "implement", "derive"]
                        },
                        {
                            pattern: "mismatched types",
                            category: "typing",
                            severity: "medium",
                            solutions: [
                                "Check type annotations",
                                "Use type conversion methods",
                                "Verify function return types"
                            ],
                            keywords: ["type", "mismatch", "convert"]
                        }
                    ]
                },
                go: {
                    common_failures: [
                        {
                            pattern: "no required module provides package",
                            category: "dependency",
                            severity: "high",
                            solutions: [
                                "Run: go mod tidy",
                                "Add dependency: go get <package>",
                                "Check import path spelling"
                            ],
                            keywords: ["module", "package", "import"]
                        },
                        {
                            pattern: "undefined:",
                            category: "scope",
                            severity: "high",
                            solutions: [
                                "Check if identifier is exported (capitalized)",
                                "Add appropriate import",
                                "Verify package name"
                            ],
                            keywords: ["undefined", "export", "import"]
                        },
                        {
                            pattern: "cannot use.*as.*in",
                            category: "typing",
                            severity: "medium",
                            solutions: [
                                "Check type compatibility",
                                "Use type conversion",
                                "Verify interface implementation"
                            ],
                            keywords: ["type", "convert", "interface"]
                        },
                        {
                            pattern: "too many arguments",
                            category: "api",
                            severity: "medium",
                            solutions: [
                                "Check function signature",
                                "Verify API documentation",
                                "Remove extra arguments"
                            ],
                            keywords: ["arguments", "signature", "parameters"]
                        }
                    ]
                },
                build: {
                    common_failures: [
                        {
                            pattern: "command not found",
                            category: "environment",
                            severity: "high",
                            solutions: [
                                "Check if tool is installed",
                                "Verify PATH environment variable",
                                "Install missing tool via devbox"
                            ],
                            keywords: ["command", "not found", "path"]
                        },
                        {
                            pattern: "permission denied",
                            category: "permissions",
                            severity: "medium",
                            solutions: [
                                "Check file permissions: chmod +x",
                                "Verify directory access rights",
                                "Run with appropriate user"
                            ],
                            keywords: ["permission", "denied", "access"]
                        },
                        {
                            pattern: "disk.*full|no space left",
                            category: "resources",
                            severity: "critical",
                            solutions: [
                                "Clean up temporary files",
                                "Remove old build artifacts",
                                "Increase available disk space"
                            ],
                            keywords: ["disk", "space", "full"]
                        },
                        {
                            pattern: "timeout|timed out",
                            category: "performance",
                            severity: "medium",
                            solutions: [
                                "Increase timeout values",
                                "Check network connectivity",
                                "Optimize slow operations"
                            ],
                            keywords: ["timeout", "network", "slow"]
                        }
                    ]
                }
            },
            learning_config: {
                min_occurrences_for_pattern: 3,
                similarity_threshold: 0.8,
                max_pattern_length: 200,
                retention_days: 180
            }
        } | save .failures/pattern_db.json
    }
    
    if not (".failures/config.json" | path exists) {
        {
            version: "1.0",
            environments: {
                python: { path: "python-env", build_command: "devbox run lint && devbox run test" },
                typescript: { path: "typescript-env", build_command: "devbox run lint && devbox run test" },
                rust: { path: "rust-env", build_command: "devbox run lint && devbox run test" },
                go: { path: "go-env", build_command: "devbox run lint && devbox run test" },
                nushell: { path: "nushell-env", build_command: "devbox run check && devbox run test" }
            },
            analysis_config: {
                max_log_lines: 1000,
                pattern_extraction_window: 50,
                confidence_threshold: 0.7
            }
        } | save .failures/config.json
    }
}

# Record a failure with detailed context
def "failure record" [
    environment: string,
    command: string,
    exit_code: int,
    output: string,
    --category: string = "unknown",
    --context: record = {}
] {
    failure init
    
    let failure_record = {
        timestamp: (date now),
        environment: $environment,
        command: $command,
        exit_code: $exit_code,
        output: ($output | lines | first 100 | str join "\n"), # Limit output size
        category: $category,
        context: $context,
        analyzed: false,
        patterns_found: [],
        suggested_solutions: []
    }
    
    $failure_record | to json --raw | save --append .failures/failure_logs.jsonl
    
    # Trigger immediate analysis for this failure
    failure analyze-single $failure_record
}

# Analyze a single failure for patterns
def "failure analyze-single" [failure_record: record] {
    let pattern_db = (open .failures/pattern_db.json)
    
    mut found_patterns = []
    mut suggested_solutions = []
    
    # Determine which pattern set to use
    let pattern_sets = if $failure_record.environment in $pattern_db.patterns {
        [$failure_record.environment, "build"]
    } else {
        ["build"]
    }
    
    for pattern_set in $pattern_sets {
        let patterns = ($pattern_db.patterns | get $pattern_set | get common_failures)
        
        for pattern_def in $patterns {
            # Check if pattern matches the failure output
            if ($failure_record.output | str contains --regex $pattern_def.pattern) {
                let confidence = (failure calculate-confidence $failure_record.output $pattern_def)
                
                if $confidence >= 0.6 {
                    $found_patterns = ($found_patterns | append {
                        pattern: $pattern_def.pattern,
                        category: $pattern_def.category,
                        severity: $pattern_def.severity,
                        confidence: $confidence,
                        source: $pattern_set
                    })
                    
                    $suggested_solutions = ($suggested_solutions | append $pattern_def.solutions)
                }
            }
        }
    }
    
    # Remove duplicate solutions
    $suggested_solutions = ($suggested_solutions | flatten | uniq)
    
    # Update the failure record with analysis
    let analyzed_record = (
        $failure_record 
        | upsert analyzed true
        | upsert patterns_found $found_patterns
        | upsert suggested_solutions $suggested_solutions
        | upsert analysis_timestamp (date now)
    )
    
    if ($found_patterns | length) > 0 {
        print $"🔍 Failure pattern detected in ($failure_record.environment):"
        $found_patterns | each { |pattern|
            print $"  • ($pattern.category): ($pattern.pattern) (confidence: ($pattern.confidence * 100 | math round --precision 1)%)"
        }
        
        print $"\n💡 Suggested solutions:"
        $suggested_solutions | each { |solution|
            print $"  • ($solution)"
        }
    }
    
    $analyzed_record
}

# Calculate confidence score for pattern match
def "failure calculate-confidence" [output: string, pattern_def: record] {
    let base_confidence = 0.6  # Base confidence for regex match
    
    # Count keyword matches for additional confidence
    let keyword_matches = ($pattern_def.keywords | each { |keyword|
        if ($output | str downcase | str contains ($keyword | str downcase)) { 1 } else { 0 }
    } | math sum)
    
    let keyword_bonus = ($keyword_matches / ($pattern_def.keywords | length) * 0.3)
    
    # Context-based confidence boost
    let context_bonus = if ($output | str length) > 100 { 0.1 } else { 0 }
    
    ($base_confidence + $keyword_bonus + $context_bonus) | math min 1.0
}

# Analyze historical failures to learn new patterns
def "failure learn-patterns" [--days: int = 30] {
    failure init
    
    let start_date = (date now) - ($days * 1day)
    let config = (open .failures/config.json)
    
    print $"🧠 Learning from failure patterns over the last ($days) days..."
    
    let failures = (
        open .failures/failure_logs.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $start_date
        | where analyzed == false or patterns_found == []
    )
    
    if ($failures | length) == 0 {
        print "No unanalyzed failures found for learning"
        return
    }
    
    print $"  Analyzing ($failures | length) unanalyzed failures..."
    
    # Group failures by environment and error patterns
    let failure_groups = (
        $failures
        | group-by environment
        | transpose environment failures
        | each { |group|
            let common_errors = (
                $group.failures
                | each { |failure|
                    # Extract error lines (typically contain keywords like "error", "failed", "exception")
                    $failure.output 
                    | lines 
                    | where ($it | str downcase | str contains --regex "(error|failed|exception|traceback|panic)")
                    | first 3
                }
                | flatten
                | where ($it | str length) > 10
                | group-by { |line| $line }
                | transpose error_line occurrences
                | where ($it.occurrences | length) >= $config.learning_config.min_occurrences_for_pattern
                | each { |common_error|
                    {
                        pattern: ($common_error.error_line | str replace --regex "[0-9]+|/[^\\s]*" ".*"), # Generalize numbers and paths
                        occurrences: ($common_error.occurrences | length),
                        category: (failure categorize-error $common_error.error_line),
                        severity: (failure assess-severity $common_error.error_line),
                        examples: ($common_error.occurrences | first 2)
                    }
                }
            )
            
            {
                environment: $group.environment,
                new_patterns: $common_errors
            }
        }
    )
    
    # Update pattern database with learned patterns
    mut pattern_db = (open .failures/pattern_db.json)
    mut learned_count = 0
    
    for group in $failure_groups {
        if ($group.new_patterns | length) > 0 {
            print $"  📚 Learned ($group.new_patterns | length) new patterns for ($group.environment):"
            
            for new_pattern in $group.new_patterns {
                print $"    • ($new_pattern.category): ($new_pattern.pattern) (($new_pattern.occurrences) occurrences)"
                
                # Add to pattern database
                let current_patterns = ($pattern_db.patterns | get $group.environment | get common_failures)
                let updated_patterns = ($current_patterns | append {
                    pattern: $new_pattern.pattern,
                    category: $new_pattern.category,
                    severity: $new_pattern.severity,
                    solutions: [(failure suggest-solution $new_pattern.pattern $new_pattern.category)],
                    keywords: (failure extract-keywords $new_pattern.pattern),
                    learned: true,
                    learn_date: (date now),
                    occurrences: $new_pattern.occurrences
                })
                
                $pattern_db = ($pattern_db | upsert patterns.($group.environment).common_failures $updated_patterns)
                $learned_count = ($learned_count + 1)
            }
        }
    }
    
    if $learned_count > 0 {
        $pattern_db | save .failures/pattern_db.json
        print $"✅ Learned ($learned_count) new failure patterns"
    } else {
        print "No new patterns learned from recent failures"
    }
}

# Categorize error type based on content
def "failure categorize-error" [error_line: string] {
    let categories = {
        dependency: ["module", "import", "package", "cannot find", "not found"],
        syntax: ["syntax", "invalid", "unexpected", "parse"],
        typing: ["type", "argument", "parameter", "assignable"],
        environment: ["command not found", "permission", "path"],
        resources: ["memory", "disk", "space", "timeout"],
        network: ["connection", "timeout", "network", "dns"],
        build: ["compilation", "build", "make", "cmake"]
    }
    
    for category in ($categories | transpose name keywords) {
        let matches = ($category.keywords | each { |keyword|
            if ($error_line | str downcase | str contains $keyword) { 1 } else { 0 }
        } | math sum)
        
        if $matches > 0 {
            return $category.name
        }
    }
    
    "unknown"
}

# Assess error severity based on content
def "failure assess-severity" [error_line: string] {
    let critical_indicators = ["panic", "segmentation fault", "out of memory", "disk full"]
    let high_indicators = ["error", "failed", "exception", "not found"]
    let medium_indicators = ["warning", "deprecated", "mismatch"]
    
    if ($critical_indicators | any { |indicator| $error_line | str downcase | str contains $indicator }) {
        "critical"
    } else if ($high_indicators | any { |indicator| $error_line | str downcase | str contains $indicator }) {
        "high"
    } else if ($medium_indicators | any { |indicator| $error_line | str downcase | str contains $indicator }) {
        "medium"
    } else {
        "low"
    }
}

# Suggest solution based on pattern and category
def "failure suggest-solution" [pattern: string, category: string] {
    let generic_solutions = {
        dependency: "Check dependencies and installation",
        syntax: "Review syntax and formatting",
        typing: "Verify types and function signatures",
        environment: "Check environment setup and permissions",
        resources: "Monitor and optimize resource usage",
        network: "Check network connectivity and timeouts",
        build: "Review build configuration and dependencies"
    }
    
    $generic_solutions | get $category | default "Review error details and consult documentation"
}

# Extract keywords from pattern for better matching
def "failure extract-keywords" [pattern: string] {
    $pattern 
    | str replace --all --regex "[^a-zA-Z ]" " "
    | split row " "
    | where ($it | str length) > 2
    | each { |word| $word | str downcase }
    | uniq
    | first 5
}

# Generate failure analysis report
def "failure report" [
    --days: int = 7,
    --environment: string = "all",
    --category: string = "all"
] {
    failure init
    
    let start_date = (date now) - ($days * 1day)
    
    let failures = (
        open .failures/failure_logs.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $start_date
    )
    
    let filtered_failures = if $environment == "all" {
        $failures
    } else {
        $failures | where environment == $environment
    }
    
    let final_failures = if $category == "all" {
        $filtered_failures
    } else {
        $filtered_failures | where category == $category
    }
    
    print $"🚨 Failure Analysis Report - Last ($days) days"
    print "=" * 50
    
    if ($final_failures | length) == 0 {
        print "No failures recorded in the specified period"
        return
    }
    
    print $"Total failures: ($final_failures | length)"
    
    # Failure summary by environment
    let env_summary = (
        $final_failures
        | group-by environment
        | transpose environment failures
        | each { |row|
            {
                environment: $row.environment,
                count: ($row.failures | length),
                success_rate: 0, # TODO: Calculate based on total attempts
                patterns_identified: ($row.failures | where analyzed == true | length)
            }
        }
        | sort-by count --reverse
    )
    
    print "\nFailure Summary by Environment:"
    $env_summary | table
    
    # Most common failure patterns
    let pattern_summary = (
        $final_failures
        | where patterns_found != []
        | each { |failure|
            $failure.patterns_found | each { |pattern|
                {
                    pattern: $pattern.pattern,
                    category: $pattern.category,
                    severity: $pattern.severity,
                    environment: $failure.environment
                }
            }
        }
        | flatten
        | group-by pattern
        | transpose pattern occurrences
        | each { |row|
            {
                pattern: $row.pattern,
                count: ($row.occurrences | length),
                category: ($row.occurrences | first | get category),
                severity: ($row.occurrences | first | get severity),
                environments: ($row.occurrences | get environment | uniq)
            }
        }
        | sort-by count --reverse
    )
    
    if ($pattern_summary | length) > 0 {
        print "\nMost Common Failure Patterns:"
        $pattern_summary | first 10 | each { |pattern|
            print $"  • ($pattern.category)/($pattern.severity): ($pattern.count) occurrences"
            print $"    Pattern: ($pattern.pattern)"
            print $"    Environments: ($pattern.environments | str join ', ')"
            print ""
        }
    }
    
    # Recent critical failures
    let critical_failures = (
        $final_failures
        | where patterns_found != []
        | where { |failure|
            $failure.patterns_found | any { |pattern| $pattern.severity == "critical" }
        }
        | sort-by timestamp --reverse
        | first 5
    )
    
    if ($critical_failures | length) > 0 {
        print "🚨 Recent Critical Failures:"
        $critical_failures | each { |failure|
            print $"  • ($failure.environment) at (($failure.timestamp | format date '%m-%d %H:%M'))"
            let critical_pattern = ($failure.patterns_found | where severity == "critical" | first)
            print $"    ($critical_pattern.category): ($critical_pattern.pattern)"
            print $"    Solutions: ($failure.suggested_solutions | str join '; ')"
            print ""
        }
    }
    
    # Learning opportunities
    let unanalyzed = ($final_failures | where analyzed == false | length)
    if $unanalyzed > 0 {
        print $"💡 Learning Opportunities: ($unanalyzed) unanalyzed failures"
        print "   Run 'failure learn-patterns' to extract new patterns"
    }
}

# Simulate and test failure scenarios
def "failure simulate" [
    environment: string,
    --scenario: string = "dependency_missing"
] {
    print $"🧪 Simulating failure scenario: ($scenario) in ($environment)"
    
    let scenarios = {
        dependency_missing: {
            command: "python -c 'import nonexistent_module'",
            expected_pattern: "ModuleNotFoundError",
            category: "dependency"
        },
        syntax_error: {
            command: "python -c 'print(\"hello\"'",
            expected_pattern: "SyntaxError",
            category: "syntax"
        },
        permission_denied: {
            command: "cat /etc/shadow",
            expected_pattern: "permission denied",
            category: "permissions"
        }
    }
    
    if not ($scenario in $scenarios) {
        print $"Unknown scenario: ($scenario)"
        print $"Available scenarios: ($scenarios | transpose name details | get name | str join ', ')"
        return
    }
    
    let scenario_def = ($scenarios | get $scenario)
    
    # Execute the failing command
    let result = (try {
        nu -c $scenario_def.command | complete
    } catch {
        {stdout: "", stderr: $in, exit_code: 1}
    })
    
    if $result.exit_code != 0 {
        print "✅ Failure simulated successfully"
        
        # Record the simulated failure
        failure record $environment $scenario_def.command $result.exit_code $result.stderr --category $scenario_def.category --context {simulated: true, scenario: $scenario}
        
        print "Failure recorded and analyzed"
    } else {
        print "❌ Failed to simulate failure - command succeeded unexpectedly"
    }
}

# Clean old failure data
def "failure cleanup" [--days: int = 180] {
    failure init
    
    let cutoff_date = (date now) - ($days * 1day)
    
    let current_failures = (
        open .failures/failure_logs.jsonl
        | lines
        | each { |line| $line | from json }
        | where timestamp > $cutoff_date
    )
    
    $current_failures | each { |failure| $failure | to json --raw } | str join "\n" | save .failures/failure_logs.jsonl
    
    print $"🧹 Cleaned failure data older than ($days) days"
}

# Export failure patterns for sharing
def "failure export" [--format: string = "json"] {
    let pattern_db = (open .failures/pattern_db.json)
    let export_data = {
        exported_at: (date now),
        version: $pattern_db.version,
        patterns: $pattern_db.patterns,
        metadata: {
            total_environments: ($pattern_db.patterns | transpose name data | length),
            total_patterns: (
                $pattern_db.patterns 
                | transpose name data 
                | each { |env| $env.data.common_failures | length } 
                | math sum
            )
        }
    }
    
    match $format {
        "json" => {
            $export_data | to json
        },
        "yaml" => {
            $export_data | to yaml
        },
        _ => {
            print "Invalid format. Use: json, yaml"
        }
    }
}

# Main command dispatcher
def main [command: string, ...args] {
    match $command {
        "init" => { failure init },
        "record" => {
            if ($args | length) >= 4 {
                failure record $args.0 $args.1 ($args.2 | into int) $args.3 ...(($args | skip 4))
            } else {
                print "Usage: failure record <environment> <command> <exit_code> <output> [--category category]"
            }
        },
        "analyze-single" => {
            if ($args | length) >= 1 {
                failure analyze-single ($args.0 | from json)
            } else {
                print "Usage: failure analyze-single <failure_record_json>"
            }
        },
        "learn-patterns" => { failure learn-patterns ...$args },
        "report" => { failure report ...$args },
        "simulate" => {
            if ($args | length) >= 1 {
                failure simulate $args.0 ...(($args | skip 1))
            } else {
                print "Usage: failure simulate <environment> [--scenario name]"
            }
        },
        "cleanup" => { failure cleanup ...$args },
        "export" => { failure export ...$args },
        _ => {
            print "Failure Pattern Learning and Analysis System"
            print "Usage:"
            print "  failure init                           - Initialize failure pattern learning"
            print "  failure record <env> <cmd> <code> <output> - Record a failure for analysis"
            print "  failure learn-patterns [--days N]     - Learn new patterns from historical failures"
            print "  failure report [--days N] [--env]     - Generate failure analysis report"
            print "  failure simulate <env> [--scenario]   - Simulate failure scenarios for testing"
            print "  failure cleanup [--days N]            - Clean old failure data"
            print "  failure export [--format json|yaml]   - Export patterns for sharing"
        }
    }
}