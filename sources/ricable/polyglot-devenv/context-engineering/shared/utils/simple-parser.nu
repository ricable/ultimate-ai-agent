#!/usr/bin/env nu
# Simplified argument parser for context engineering commands

# Parse command arguments with basic support
def "parse arguments" [
    raw_args: string,              # Raw argument string from $ARGUMENTS
    command_spec: record           # Command specification with expected arguments
] {
    
    if ($raw_args | str contains "--help") or ($raw_args | str contains "-h") {
        print_help $command_spec
        return {parsed: {}, help_shown: true}
    }
    
    # Split arguments
    let args = if ($raw_args | str trim) == "" { [] } else { ($raw_args | split row ' ' | where $it != "") }
    
    mut parsed = {}
    mut positional = []
    mut i = 0
    
    # Process arguments
    while $i < ($args | length) {
        let arg = ($args | get $i)
        
        if ($arg | str starts-with '--') {
            # Handle long flags
            let flag_name = ($arg | str replace '--' '')
            if ($i + 1) < ($args | length) {
                let next_arg = ($args | get ($i + 1))
                if not ($next_arg | str starts-with '-') {
                    $parsed = ($parsed | insert $flag_name $next_arg)
                    $i = $i + 2
                } else {
                    $parsed = ($parsed | insert $flag_name true)
                    $i = $i + 1
                }
            } else {
                $parsed = ($parsed | insert $flag_name true)
                $i = $i + 1
            }
        } else if ($arg | str starts-with '-') {
            # Handle short flags
            let flag_name = ($arg | str replace '-' '')
            let full_name = (get_flag_full_name $flag_name $command_spec)
            if ($i + 1) < ($args | length) {
                let next_arg = ($args | get ($i + 1))
                if not ($next_arg | str starts-with '-') {
                    $parsed = ($parsed | insert $full_name $next_arg)
                    $i = $i + 2
                } else {
                    $parsed = ($parsed | insert $full_name true)
                    $i = $i + 1
                }
            } else {
                $parsed = ($parsed | insert $full_name true)
                $i = $i + 1
            }
        } else {
            # Positional argument
            $positional = ($positional | append $arg)
            $i = $i + 1
        }
    }
    
    # Add positional arguments
    if (($positional | length) > 0) {
        $parsed = ($parsed | insert "_positional" $positional)
    }
    
    # Apply defaults
    if ($command_spec.arguments? != null) {
        for spec in $command_spec.arguments {
            if ($spec.name not-in ($parsed | columns)) and ($spec.default? != null) {
                $parsed = ($parsed | insert $spec.name $spec.default)
            }
        }
    }
    
    return {parsed: $parsed, help_shown: false, error: null}
}

# Get full flag name from short flag
def get_flag_full_name [
    short_flag: string,           # Short flag (e, t, h)
    command_spec: record          # Command specification
] {
    if ($command_spec.arguments? != null) {
        for spec in $command_spec.arguments {
            if ($spec.short? == $short_flag) {
                return $spec.name
            }
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
    
    if ($command_spec.arguments? != null) and (($command_spec.arguments | length) > 0) {
        print "📋 Arguments:"
        
        for spec in $command_spec.arguments {
            let required_mark = if ($spec.required? == true) { "*" } else { " " }
            print $"   ($required_mark) --($spec.name)"
            print $"     ($spec.description)"
            if ($spec.short? != null) {
                print $"     Short: -($spec.short)"
            }
            if ($spec.choices? != null) {
                print $"     Choices: ($spec.choices | str join ', ')"
            }
            if ($spec.default? != null) {
                print $"     Default: ($spec.default)"
            }
        }
        print ""
        print "* = required argument"
        print ""
    }
    
    if ($command_spec.examples? != null) and (($command_spec.examples | length) > 0) {
        print "💡 Examples:"
        for example in $command_spec.examples {
            print $"   ($example)"
        }
        print ""
    }
}

# Check if arguments contain help flag
def "has help flag" [
    raw_args: string              # Raw argument string
] {
    return (($raw_args | str contains '--help') or ($raw_args | str contains '-h'))
}

# Parse list argument (comma-separated values)
def "parse list arg" [
    value: string                 # Comma-separated string
] {
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
] {
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
    --required,                   # Whether argument is required
    --default: any,               # Default value
    --choices: list,              # Valid choices
    --min: int,                   # Minimum value (for numbers)
    --max: int                    # Maximum value (for numbers)
] {
    mut spec = {
        name: $name,
        description: $description,
        required: ($required != null)
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
    print "Enhanced Context Engineering Argument Parser"
    print "Usage: source this file in your commands and use 'parse arguments'"
}