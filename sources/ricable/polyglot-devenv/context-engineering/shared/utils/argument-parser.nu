#!/usr/bin/env nu
# Enhanced argument parser for context engineering commands
# Provides structured argument parsing, validation, and help system

# Parse command arguments with support for named parameters, flags, and positional args
def "parse arguments" [
    raw_args: string,              # Raw argument string from $ARGUMENTS
    command_spec: record,          # Command specification with expected arguments
    --help                         # Show help information
]: record -> record {
    
    if $help {
        print_help $command_spec
        return {parsed: {}, help_shown: true}
    }
    
    # Split arguments while preserving quoted strings
    let args = ($raw_args | split row ' ' | where $it != "")
    
    mut parsed = {}
    mut positional = []
    mut current_flag = null
    
    # Process arguments
    for arg in $args {
        if ($arg | str starts-with '--') {
            # Handle long flags (--env python-env)
            let flag_name = ($arg | str replace '--' '')
            $current_flag = $flag_name
            $parsed = ($parsed | insert $flag_name null)
        } else if ($arg | str starts-with '-') {
            # Handle short flags (-e python-env)
            let flag_name = ($arg | str replace '-' '')
            let full_name = (get_flag_full_name $flag_name $command_spec)
            $current_flag = $full_name
            $parsed = ($parsed | insert $full_name null)
        } else {
            # Handle values
            if ($current_flag != null) {
                $parsed = ($parsed | insert $current_flag $arg)
                $current_flag = null
            } else {
                $positional = ($positional | append $arg)
            }
        }
    }
    
    # Add positional arguments to parsed result
    if (($positional | length) > 0) {
        $parsed = ($parsed | insert "_positional" $positional)
    }
    
    # Apply defaults from command spec
    for spec in ($command_spec.arguments? | default []) {
        if ($spec.name not-in ($parsed | columns)) and ($spec.default? != null) {
            $parsed = ($parsed | insert $spec.name $spec.default)
        }
    }
    
    # Validate arguments
    let validation_result = (validate_arguments $parsed $command_spec)
    if not $validation_result.valid {
        print $"❌ Validation Error: ($validation_result.error)"
        print_help $command_spec
        return {parsed: {}, error: $validation_result.error}
    }
    
    return {parsed: $parsed, help_shown: false}
}

# Validate parsed arguments against command specification
def validate_arguments [
    parsed: record,               # Parsed arguments
    command_spec: record          # Command specification
]: record -> record {
    
    let required_args = ($command_spec.arguments? | default [] | where required == true)
    
    for req_arg in $required_args {
        if ($req_arg.name not-in ($parsed | columns)) {
            return {valid: false, error: $"Required argument '--($req_arg.name)' is missing"}
        }
        
        let value = ($parsed | get $req_arg.name)
        if ($value == null) or ($value == "") {
            return {valid: false, error: $"Required argument '--($req_arg.name)' cannot be empty"}
        }
    }
    
    # Validate argument types and constraints
    for spec in ($command_spec.arguments? | default []) {
        if ($spec.name in ($parsed | columns)) {
            let value = ($parsed | get $spec.name)
            
            # Type validation
            if ($spec.type? != null) and ($value != null) {
                let validation = (validate_type $value $spec.type)
                if not $validation.valid {
                    return {valid: false, error: $"Argument '--($spec.name)': ($validation.error)"}
                }
            }
            
            # Choice validation
            if ($spec.choices? != null) and ($value != null) {
                if ($value not-in $spec.choices) {
                    let choices_str = ($spec.choices | str join ", ")
                    return {valid: false, error: $"Argument '--($spec.name)' must be one of: ($choices_str)"}
                }
            }
            
            # Range validation for numbers
            if ($spec.min? != null) and ($value != null) {
                let num_value = ($value | into int)
                if ($num_value < $spec.min) {
                    return {valid: false, error: $"Argument '--($spec.name)' must be at least ($spec.min)"}
                }
            }
            
            if ($spec.max? != null) and ($value != null) {
                let num_value = ($value | into int)
                if ($num_value > $spec.max) {
                    return {valid: false, error: $"Argument '--($spec.name)' cannot exceed ($spec.max)"}
                }
            }
        }
    }
    
    return {valid: true, error: null}
}

# Validate argument type
def validate_type [
    value: any,                   # Value to validate
    expected_type: string         # Expected type (string, int, bool, list)
] -> record {
    
    match $expected_type {
        "string" => {
            if ($value | describe | str starts-with "string") {
                return {valid: true, error: null}
            } else {
                return {valid: false, error: "must be a string"}
            }
        }
        "int" => {
            try {
                $value | into int
                return {valid: true, error: null}
            } catch {
                return {valid: false, error: "must be an integer"}
            }
        }
        "bool" => {
            if ($value in ["true", "false", "yes", "no", "1", "0"]) {
                return {valid: true, error: null}
            } else {
                return {valid: false, error: "must be true/false, yes/no, or 1/0"}
            }
        }
        "list" => {
            if ($value | str contains ',') {
                return {valid: true, error: null}
            } else {
                return {valid: false, error: "must be a comma-separated list"}
            }
        }
        _ => {
            return {valid: true, error: null}
        }
    }
}

# Get full flag name from short flag
def get_flag_full_name [
    short_flag: string,           # Short flag (e, t, h)
    command_spec: record          # Command specification
] -> string {
    
    for spec in ($command_spec.arguments? | default []) {
        if ($spec.short? == $short_flag) {
            return $spec.name
        }
    }
    
    return $short_flag
}

# Print help information for a command
def print_help [
    command_spec: record          # Command specification
] {
    print $"📖 ($command_spec.name)"
    print $"   ($command_spec.description)"
    print ""
    
    if ($command_spec.usage? != null) {
        print $"🔧 Usage: ($command_spec.usage)"
        print ""
    }
    
    if (($command_spec.arguments? | default [] | length) > 0) {
        print "📋 Arguments:"
        
        for spec in $command_spec.arguments {
            let required_mark = if ($spec.required? == true) { "*" } else { " " }
            let short_flag = if ($spec.short? != null) { $" (-($spec.short))" } else { "" }
            let default_text = if ($spec.default? != null) { $" (default: ($spec.default))" } else { "" }
            let choices_text = if ($spec.choices? != null) { $" [($spec.choices | str join '|')]" } else { "" }
            
            print $"   ($required_mark) --($spec.name)($short_flag)($choices_text)($default_text)"
            print $"     ($spec.description)"
        }
        print ""
        print "* = required argument"
        print ""
    }
    
    if (($command_spec.examples? | default [] | length) > 0) {
        print "💡 Examples:"
        for example in $command_spec.examples {
            print $"   ($example)"
        }
        print ""
    }
}

# Convert parsed arguments to environment variables format
def "args to env" [
    parsed: record                # Parsed arguments
] -> record {
    
    mut env_vars = {}
    
    for key in ($parsed | columns) {
        if ($key != "_positional") {
            let env_key = ($key | str upcase | str replace '-' '_')
            let value = ($parsed | get $key)
            $env_vars = ($env_vars | insert $env_key $value)
        }
    }
    
    return $env_vars
}

# Check if arguments contain help flag
def "has help flag" [
    raw_args: string              # Raw argument string
] -> bool {
    
    return (($raw_args | str contains '--help') or ($raw_args | str contains '-h'))
}

# Parse list argument (comma-separated values)
def "parse list arg" [
    value: string                 # Comma-separated string
] -> list {
    
    if ($value | str trim) == "" {
        return []
    }
    
    return ($value | split row ',' | each { |item| $item | str trim })
}

# Create command specification record
def "create command spec" [
    name: string,                 # Command name
    description: string,          # Command description
    arguments: list,              # List of argument specifications
    --usage: string,              # Usage string
    --examples: list              # List of example commands
] -> record {
    
    mut spec = {
        name: $name,
        description: $description,
        arguments: $arguments
    }
    
    if ($usage != null) {
        $spec = ($spec | insert usage $usage)
    }
    
    if ($examples != null) {
        $spec = ($spec | insert examples $examples)
    }
    
    return $spec
}

# Create argument specification record
def "create arg spec" [
    name: string,                 # Argument name
    description: string,          # Argument description
    --type: string,               # Argument type (string, int, bool, list)
    --short: string,              # Short flag (-e for --env)
    --required: bool = false,     # Whether argument is required
    --default: any,               # Default value
    --choices: list,              # Valid choices
    --min: int,                   # Minimum value (for numbers)
    --max: int                    # Maximum value (for numbers)
] -> record {
    
    mut spec = {
        name: $name,
        description: $description,
        required: $required
    }
    
    if ($type != null) { $spec = ($spec | insert type $type) }
    if ($short != null) { $spec = ($spec | insert short $short) }
    if ($default != null) { $spec = ($spec | insert default $default) }
    if ($choices != null) { $spec = ($spec | insert choices $choices) }
    if ($min != null) { $spec = ($spec | insert min $min) }
    if ($max != null) { $spec = ($spec | insert max $max) }
    
    return $spec
}

# Export main functions for use in commands
export def main [] {
    print "Context Engineering Argument Parser"
    print "Usage: source this file in your commands and use 'parse arguments'"
}