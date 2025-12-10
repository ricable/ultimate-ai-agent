#!/usr/bin/env nu

# Simple test of the configuration generation concept

def main [] {
    log info "🧪 Testing Configuration Generation Concept"
    
    # Define a simple Python environment for testing
    let python_config = {
        name: "Python Development Environment",
        packages: {
            devbox: ["python@3.12", "uv", "ruff", "mypy", "nushell"]
        },
        environment: {
            PYTHONPATH: "$PWD/src",
            UV_CACHE_DIR: "$PWD/.uv-cache"
        },
        scripts: {
            format: "uv run ruff format .",
            lint: "uv run ruff check . --fix",
            test: "uv run pytest --cov=src"
        }
    }
    
    # Generate devbox.json
    let devbox_config = {
        packages: $python_config.packages.devbox,
        shell: {
            init_hook: [
                $"echo '($python_config.name)'",
                "uv --version",
                "python --version"
            ],
            scripts: $python_config.scripts
        },
        env: $python_config.environment
    }
    
    log success "✅ Generated devbox.json structure:"
    print ($devbox_config | to json --indent 2)
    
    log info "💡 This demonstrates the single source of truth concept"
    log info "   - One definition generates multiple config formats"
    log info "   - No duplication between dev-env/ and devpod-automation/"
    log info "   - Changes in one place propagate everywhere"
}

def log [level: string, message: string] {
    let timestamp = (date now | format date "%H:%M:%S")
    match $level {
        "info" => { print $"[($timestamp)] ℹ️  ($message)" }
        "success" => { print $"[($timestamp)] ✅ ($message)" }
        "error" => { print $"[($timestamp)] ❌ ($message)" }
        _ => { print $"[($timestamp)] ($message)" }
    }
}