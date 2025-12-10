#!/usr/bin/env nu

# Common utilities and functions for the polyglot development environment
# This module provides shared functionality across all Nushell scripts

# Environment variable management
export def "env get-or-prompt" [
    var_name: string
    prompt_text: string
    --secret = false
] {
    if ($var_name in $env) {
        $env | get $var_name
    } else {
        let value = if $secret {
            input $"(ansi yellow_bold)($prompt_text): (ansi reset)" --suppress-output
        } else {
            input $"(ansi yellow_bold)($prompt_text): (ansi reset)"
        }
        
        $"export ($var_name)=($value)\n" | save --append .env
        $value
    }
}

# Safe environment variable setting
export def "env set-safe" [
    var_name: string
    value: string
] {
    $env | upsert $var_name $value
    $"export ($var_name)=($value)\n" | save --append .env
}

# Colored logging functions
export def "log info" [message: string] {
    print $"(ansi blue_bold)[INFO](ansi reset) ($message)"
}

export def "log warn" [message: string] {
    print $"(ansi yellow_bold)[WARN](ansi reset) ($message)"
}

export def "log error" [message: string] {
    print $"(ansi red_bold)[ERROR](ansi reset) ($message)"
}

export def "log success" [message: string] {
    print $"(ansi green_bold)[SUCCESS](ansi reset) ($message)"
}

# Command execution with error handling
export def "run safe" [
    command: string
    --ignore-errors = false
] {
    if $ignore_errors {
        do --ignore-errors { bash -c $command }
    } else {
        bash -c $command
    }
}

# Check if command exists
export def "cmd exists" [command: string] {
    (which $command | length) > 0
}

# Wait for condition with timeout
export def "wait for" [
    condition: closure
    --timeout = 60
    --interval = 2
] {
    mut attempts = 0
    let max_attempts = ($timeout / $interval)
    
    while $attempts < $max_attempts {
        if (do $condition) {
            return true
        }
        sleep ($interval * 1sec)
        $attempts = $attempts + 1
    }
    
    false
}

# JSON/YAML configuration helpers
export def "config load" [file_path: string] {
    if ($file_path | path exists) {
        if ($file_path | str ends-with ".json") {
            open $file_path | from json
        } else if (($file_path | str ends-with ".yaml") or ($file_path | str ends-with ".yml")) {
            open $file_path | from yaml
        } else {
            open $file_path
        }
    } else {
        {}
    }
}

export def "config save" [
    data: any
    file_path: string
] {
    if ($file_path | str ends-with ".json") {
        $data | to json | save $file_path --force
    } else if (($file_path | str ends-with ".yaml") or ($file_path | str ends-with ".yml")) {
        $data | to yaml | save $file_path --force
    } else {
        $data | save $file_path --force
    }
}

# Development environment helpers
export def "dev setup-all" [] {
    let environments = ["dev-env/python", "dev-env/typescript", "dev-env/rust", "dev-env/go"]
    
    for env in $environments {
        if ($env | path exists) {
            log info $"Setting up ($env)..."
            cd $env
            devbox run install
            cd ..
            log success $"($env) setup completed"
        } else {
            log warn $"($env) not found, skipping..."
        }
    }
}

export def "dev test-all" [] {
    let environments = [
        {name: "Python", dir: "dev-env/python", commands: ["lint", "test"]},
        {name: "TypeScript", dir: "dev-env/typescript", commands: ["lint", "test"]}, 
        {name: "Rust", dir: "dev-env/rust", commands: ["lint", "test"]},
        {name: "Go", dir: "dev-env/go", commands: ["lint", "test"]},
        {name: "Nushell", dir: "dev-env/nushell", commands: ["check", "test"]}
    ]
    
    for env in $environments {
        if ($env.dir | path exists) {
            log info $"Testing ($env.name)..."
            cd $env.dir
            for cmd in $env.commands {
                devbox run $cmd
            }
            cd ..
            log success $"($env.name) tests passed"
        } else {
            log warn $"($env.dir) not found, skipping ($env.name)..."
        }
    }
}

# Git helpers
export def "git current-branch" [] {
    git branch --show-current
}

export def "git is-clean" [] {
    (git status --porcelain | lines | length) == 0
}

export def "git commit-safe" [message: string] {
    if not (git is-clean) {
        git add .
        git commit -m $message
        log success $"Committed: ($message)"
    } else {
        log info "No changes to commit"
    }
}

# File system helpers
export def "fs backup" [file_path: string] {
    let backup_path = $"($file_path).backup.(date now | format date '%Y%m%d_%H%M%S')"
    cp $file_path $backup_path
    log info $"Backup created: ($backup_path)"
    $backup_path
}

export def "fs cleanup-backups" [
    pattern: string = "*.backup.*"
    --keep = 5
] {
    ls $pattern 
    | sort-by modified 
    | reverse 
    | skip $keep 
    | each { |file| 
        rm $file.name
        log info $"Removed old backup: ($file.name)"
    }
}

# Network helpers
export def "net wait-for-port" [
    port: int
    host: string = "localhost"
    --timeout = 30
] {
    wait for {
        do --ignore-errors { 
            http get $"http://($host):($port)" 
        } | is-not-empty
    } --timeout $timeout
}

# Secret management integration
export def "secret get" [key: string] {
    if (cmd exists "teller") {
        teller get $key
    } else {
        env get-or-prompt $key $"Enter value for ($key)" --secret true
    }
}