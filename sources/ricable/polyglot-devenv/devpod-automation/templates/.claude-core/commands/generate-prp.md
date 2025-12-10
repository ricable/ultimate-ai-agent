# /generate-prp

Enhanced PRP generation with dynamic templates, smart dojo integration, and context-aware analysis.

## Usage

```bash
/generate-prp <feature-file> [options]
```

## Arguments

- `--env, -e`: Target environment Choices: `python-env, typescript-env, rust-env, go-env, nushell-env, multi`
- `--template, -t`: Template to use (auto-detected if not specified)
- `--examples`: Examples to include (comma-separated, auto-detected if not specified)
- `--include-dojo`: Include dojo patterns analysis Default: `auto`
- `--format`: Output format Default: `markdown` Choices: `markdown, json, yaml`
- `--output, -o`: Output file (auto-generated if not specified)
- `--analyze-only`: Only analyze without generating PRP Default: `false`
- `--verbose, -v`: Enable verbose output Default: `false`

## Examples

- `/generate-prp features/chat-ui.md --env typescript-env --include-dojo`
- `/generate-prp features/user-api.md --env python-env --examples user-management`
- `/generate-prp features/multi-agent.md --env multi --verbose`
- `/generate-prp features/cli-tool.md --analyze-only`

## Command Implementation

```bash
#!/usr/bin/env nu

# Source all enhanced utilities
source /Users/cedric/dev/github.com/polyglot-devenv/context-engineering/shared/utils/argument-parser.nu
source /Users/cedric/dev/github.com/polyglot-devenv/context-engineering/shared/utils/template-engine.nu
source /Users/cedric/dev/github.com/polyglot-devenv/context-engineering/shared/utils/dojo-integrator.nu

# Get raw arguments from Claude Code
let RAW_ARGS = "${ARGUMENTS}"

# Define enhanced command specification
let command_spec = {
    name: "generate-prp",
    description: "Enhanced PRP generation with dynamic templates and smart analysis",
    arguments: [
        {
            name: "env",
            description: "Target environment",
            type: "string",
            short: "e",
            choices: ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env", "multi"]
        },
        {
            name: "template",
            description: "Template to use (auto-detected if not specified)",
            type: "string",
            short: "t"
        },
        {
            name: "examples",
            description: "Examples to include (comma-separated, auto-detected if not specified)",
            type: "list"
        },
        {
            name: "include-dojo",
            description: "Include dojo patterns analysis",
            type: "string",
            default: "auto",
            choices: ["auto", "yes", "no"]
        },
        {
            name: "format",
            description: "Output format",
            type: "string",
            default: "markdown",
            choices: ["markdown", "json", "yaml"]
        },
        {
            name: "output",
            description: "Output file (auto-generated if not specified)",
            type: "string",
            short: "o"
        },
        {
            name: "analyze-only",
            description: "Only analyze without generating PRP",
            type: "bool",
            default: false
        },
        {
            name: "verbose",
            description: "Enable verbose output",
            type: "bool",
            default: false,
            short: "v"
        }
    ]
}

# Check for help flag
if (has help flag $RAW_ARGS) {
    parse arguments $RAW_ARGS $command_spec --help
    exit 0
}

# Parse arguments with enhanced validation
let result = (parse arguments $RAW_ARGS $command_spec)

if ($result.help_shown) {
    exit 0
}

if ($result.error? != null) {
    print $"❌ Error: ($result.error)"
    exit 1
}

let parsed_args = $result.parsed

# Get positional argument (feature file)
let feature_file = if ("_positional" in ($parsed_args | columns)) and (($parsed_args._positional | length) > 0) {
    $parsed_args._positional.0
} else {
    print "❌ Feature file is required as first argument"
    exit 1
}

# Validate feature file exists
if not ($feature_file | path exists) {
    print $"❌ Feature file not found: ($feature_file)"
    exit 1
}

print $"🚀 PRP Generation for: ($feature_file)"
print $"📊 Using enhanced analysis and dynamic templates..."
print ""

# Step 1: Read and analyze feature request
print "📖 Reading feature request..."
let feature_content = (open $feature_file)
if ($parsed_args.verbose? | default false) {
    print $"Feature content length: ($feature_content | str length) characters"
}

# Step 2: Auto-detect or use specified environment
let target_env = if ("env" in ($parsed_args | columns)) and ($parsed_args.env != null) {
    $parsed_args.env
} else {
    print "🔍 Auto-detecting target environment..."
    let detected_env = (auto_detect_environment $feature_content)
    print $"🎯 Detected environment: ($detected_env)"
    $detected_env
}

# Step 3: Parse dojo patterns if needed
let dojo_patterns = if (should_include_dojo $parsed_args $feature_content $target_env) {
    print "🥋 Analyzing dojo patterns..."
    let patterns = (parse dojo patterns)
    if ($parsed_args.verbose? | default false) {
        print $"Found ($patterns.features.defined_features | length) dojo features"
        print $"Component patterns: ($patterns.components.ui_components | length) UI components"
    }
    $patterns
} else {
    if ($parsed_args.verbose? | default false) {
        print "⏭️  Skipping dojo analysis (not applicable)"
    }
    {}
}

# Step 4: Determine examples to include
let examples_to_include = (determine_examples $parsed_args $feature_content $target_env $dojo_patterns)
if (($examples_to_include | length) > 0) {
    print $"📚 Including examples: ($examples_to_include | str join ', ')"
} else if ($parsed_args.verbose? | default false) {
    print "📚 No specific examples identified"
}

# Step 5: Generate dynamic template
print "🏗️  Generating dynamic template..."
let template_result = (generate dynamic template $target_env $feature_content --examples $examples_to_include)

if ($parsed_args.verbose? | default false) {
    print $"Template type: ($template_result.analysis.environment)"
    print $"Capabilities: ($template_result.analysis.capabilities | str join ', ')"
    print $"Pattern complexity: ($template_result.patterns.complexity)"
}

# Step 6: Handle analyze-only mode
if ($parsed_args.analyze-only? | default false) {
    print ""
    print "🔍 Analysis Results:"
    print "=================="
    print $"Target Environment: ($target_env)"
    print $"Feature Type: ($template_result.patterns.feature_type)"
    print $"Complexity: ($template_result.patterns.complexity)"
    print $"Examples: ($examples_to_include | str join ', ')"
    print $"Dojo Integration: (if (($dojo_patterns | columns | length) > 0) { 'Yes' } else { 'No' })"
    
    if ($parsed_args.format == "json") {
        let analysis_json = {
            environment: $target_env,
            patterns: $template_result.patterns,
            examples: $examples_to_include,
            dojo_available: (($dojo_patterns | columns | length) > 0),
            template_type: ($template_result.analysis.environment),
            capabilities: $template_result.analysis.capabilities
        }
        print ""
        print "📋 JSON Analysis:"
        print ($analysis_json | to json)
    }
    
    exit 0
}

# Step 7: Enhance template with dojo integration
let enhanced_template = if (($dojo_patterns | columns | length) > 0) {
    print "🔗 Integrating dojo patterns..."
    (integrate_dojo_patterns $template_result.template $dojo_patterns $examples_to_include $target_env)
} else {
    $template_result.template
}

# Step 8: Generate output filename
let output_file = if ("output" in ($parsed_args | columns)) and ($parsed_args.output != null) {
    $parsed_args.output
} else {
    let base_name = ($feature_file | path parse | get stem)
    let env_suffix = if ($target_env == "multi") { "multi" } else { ($target_env | str replace "-env" "") }
    $"context-engineering/PRPs/($base_name)-($env_suffix).md"
}

# Step 9: Write the enhanced PRP
print $"💾 Writing enhanced PRP to: ($output_file)"

# Ensure output directory exists
let output_dir = ($output_file | path dirname)
if not ($output_dir | path exists) {
    mkdir $output_dir
}

# Write based on format
match ($parsed_args.format) {
    "json" => {
        let json_output = {
            metadata: {
                generated_by: "generate-prp",
                timestamp: (date now | format date "%Y-%m-%d %H:%M:%S"),
                source_file: $feature_file,
                target_environment: $target_env,
                examples_included: $examples_to_include,
                dojo_integration: (($dojo_patterns | columns | length) > 0)
            },
            template: $enhanced_template,
            analysis: $template_result.analysis,
            patterns: $template_result.patterns
        }
        $json_output | to json | save $output_file
    }
    "yaml" => {
        let yaml_output = {
            metadata: {
                generated_by: "generate-prp",
                timestamp: (date now | format date "%Y-%m-%d %H:%M:%S"),
                source_file: $feature_file,
                target_environment: $target_env,
                examples_included: $examples_to_include,
                dojo_integration: (($dojo_patterns | columns | length) > 0)
            },
            template: $enhanced_template
        }
        $yaml_output | to yaml | save $output_file
    }
    _ => {
        # Markdown format (default)
        let markdown_header = $"<!-- Generated by generate-prp -->
<!-- Timestamp: (date now | format date "%Y-%m-%d %H:%M:%S") -->
<!-- Source: ($feature_file) -->
<!-- Environment: ($target_env) -->
<!-- Examples: ($examples_to_include | str join ', ') -->
<!-- Dojo Integration: (if (($dojo_patterns | columns | length) > 0) { 'Yes' } else { 'No' }) -->

"
        ($markdown_header + $enhanced_template) | save $output_file
    }
}

print ""
print "✅ PRP generation completed!"
print $"📄 Output file: ($output_file)"
print $"🎯 Environment: ($target_env)"
print $"📋 Template type: ($template_result.patterns.feature_type)"

if (($examples_to_include | length) > 0) {
    print $"📚 Examples integrated: ($examples_to_include | str join ', ')"
}

if (($dojo_patterns | columns | length) > 0) {
    print $"🥋 Dojo patterns integrated: ($dojo_patterns.features.defined_features | length) features"
}

print ""
print "💡 Next steps:"
print $"   1. Review the generated PRP: ($output_file)"
print $"   2. Execute with: /execute-prp ($output_file) --validate"
print $"   3. Or use workflow: /context workflow ($feature_file | path parse | get stem) --env ($target_env) --validate"

# Auto-detect environment from feature content
def auto_detect_environment [content: string] -> string {
    let content_lower = ($content | str downcase)
    
    if ("python" in $content_lower) or ("fastapi" in $content_lower) or ("django" in $content_lower) {
        return "python-env"
    } else if ("typescript" in $content_lower) or ("react" in $content_lower) or ("nextjs" in $content_lower) or ("node" in $content_lower) {
        return "typescript-env"
    } else if ("rust" in $content_lower) or ("cargo" in $content_lower) or ("tokio" in $content_lower) {
        return "rust-env"
    } else if ("go" in $content_lower) or ("golang" in $content_lower) or ("gin" in $content_lower) {
        return "go-env"
    } else if ("nushell" in $content_lower) or ("nu" in $content_lower) or ("script" in $content_lower) {
        return "nushell-env"
    } else if ("multi" in $content_lower) or ("cross" in $content_lower) or ("polyglot" in $content_lower) {
        return "multi"
    } else {
        # Default based on common patterns
        if ("api" in $content_lower) or ("web" in $content_lower) {
            return "python-env"  # Default for API features
        } else if ("ui" in $content_lower) or ("component" in $content_lower) or ("frontend" in $content_lower) {
            return "typescript-env"  # Default for UI features
        } else {
            return "python-env"  # Overall default
        }
    }
}

# Determine if dojo patterns should be included
def should_include_dojo [args: record, content: string, env: string] -> bool {
    let include_setting = ($args.include-dojo? | default "auto")
    
    match $include_setting {
        "yes" => { return true }
        "no" => { return false }
        "auto" => {
            # Auto-detect based on environment and content
            if ($env == "typescript-env") {
                return true  # Always include for TypeScript
            }
            
            let content_lower = ($content | str downcase)
            if ("ui" in $content_lower) or ("component" in $content_lower) or ("chat" in $content_lower) or ("copilotkit" in $content_lower) {
                return true
            }
            
            return false
        }
        _ => { return false }
    }
}

# Determine examples to include based on analysis
def determine_examples [args: record, content: string, env: string, dojo_patterns: record] -> list {
    mut examples = []
    
    # Add explicitly requested examples
    if ("examples" in ($args | columns)) and ($args.examples != null) {
        let requested = (parse list arg $args.examples)
        $examples = ($examples | append $requested)
    }
    
    # Auto-detect examples based on content
    let content_lower = ($content | str downcase)
    
    # Dojo examples based on feature type
    if (($dojo_patterns | columns | length) > 0) {
        if ("chat" in $content_lower) or ("conversation" in $content_lower) {
            $examples = ($examples | append "dojo/agentic_chat")
        }
        if ("ui" in $content_lower) or ("component" in $content_lower) {
            $examples = ($examples | append "dojo/agentic_generative_ui")
        }
        if ("collaboration" in $content_lower) or ("shared" in $content_lower) {
            $examples = ($examples | append "dojo/shared_state")
        }
        if ("human" in $content_lower) or ("approval" in $content_lower) {
            $examples = ($examples | append "dojo/human_in_the_loop")
        }
    }
    
    # Environment-specific examples
    match $env {
        "python-env" => {
            if ("api" in $content_lower) or ("rest" in $content_lower) {
                $examples = ($examples | append "python-api-example")
            }
            if ("user" in $content_lower) or ("auth" in $content_lower) {
                $examples = ($examples | append "user-management")
            }
        }
        "typescript-env" => {
            if not ("dojo" in $examples) {
                $examples = ($examples | append "dojo")  # Default for TypeScript
            }
        }
        _ => {}
    }
    
    return ($examples | uniq)
}

# Integrate dojo patterns into the template
def integrate_dojo_patterns [template: string, dojo_patterns: record, examples: list, env: string] -> string {
    
    mut enhanced_template = $template
    
    # Add dojo-specific context section
    let dojo_section = generate_dojo_context_section $dojo_patterns $examples $env
    
    # Insert dojo section after the context section
    if ($enhanced_template | str contains "## All Needed Context") {
        let parts = ($enhanced_template | split row "## All Needed Context")
        if (($parts | length) >= 2) {
            let before = $parts.0
            let after = $parts.1
            
            # Find the end of the context section
            let context_parts = ($after | split row "## Implementation Blueprint")
            if (($context_parts | length) >= 2) {
                let context_content = $context_parts.0
                let implementation_content = $context_parts.1
                
                $enhanced_template = $before + "## All Needed Context" + $context_content + $dojo_section + "## Implementation Blueprint" + $implementation_content
            }
        }
    }
    
    return $enhanced_template
}

# Generate dojo context section
def generate_dojo_context_section [dojo_patterns: record, examples: list, env: string] -> string {
    
    let relevant_features = ($examples | where ($it | str starts-with "dojo/") | each { |ex| $ex | str replace "dojo/" "" })
    
    mut dojo_context = "

### Dojo Integration Context

"
    
    if (($relevant_features | length) > 0) {
        $dojo_context = $dojo_context + "#### Relevant Dojo Features\n"
        
        for feature in $relevant_features {
            if ($feature in ($dojo_patterns.features.feature_patterns | columns)) {
                let feature_data = ($dojo_patterns.features.feature_patterns | get $feature)
                $dojo_context = $dojo_context + $"- **($feature)**: "
                
                # Add feature description from defined features
                let feature_info = ($dojo_patterns.features.defined_features | where id == $feature)
                if (($feature_info | length) > 0) {
                    $dojo_context = $dojo_context + ($feature_info.0.description? | default "CopilotKit feature pattern")
                } else {
                    $dojo_context = $dojo_context + "CopilotKit integration pattern"
                }
                
                $dojo_context = $dojo_context + "\n"
                
                # Add component patterns
                if (($feature_data.components | length) > 0) {
                    $dojo_context = $dojo_context + $"  - Components: ($feature_data.components | str join ', ')\n"
                }
                
                # Add CopilotKit usage
                if (($feature_data.copilotkit_usage | length) > 0) {
                    $dojo_context = $dojo_context + $"  - CopilotKit patterns: ($feature_data.copilotkit_usage | str join ', ')\n"
                }
            }
        }
    }
    
    # Add general dojo patterns
    if (($dojo_patterns.patterns.react_patterns | length) > 0) {
        $dojo_context = $dojo_context + "
#### Dojo React Patterns
```typescript
"
        for pattern in ($dojo_patterns.patterns.react_patterns | uniq) {
            $dojo_context = $dojo_context + $"// ($pattern)\n"
        }
        
        $dojo_context = $dojo_context + "```
"
    }
    
    # Add configuration patterns
    if (($dojo_patterns.configuration.dependencies.prod? | default {} | columns | length) > 0) {
        $dojo_context = $dojo_context + "
#### Dojo Dependencies
```json
"
        let deps = ($dojo_patterns.configuration.dependencies.prod | to json)
        $dojo_context = $dojo_context + $deps + "
```
"
    }
    
    return $dojo_context
}
```