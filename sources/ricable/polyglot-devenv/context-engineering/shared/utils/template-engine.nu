#!/usr/bin/env nu
# Dynamic Template Generation Engine for Context Engineering
# Analyzes project structure and generates custom templates

# Analyze project structure and generate dynamic template
def "generate dynamic template" [
    target_env: string,           # Target environment (python-env, typescript-env, etc.)
    feature_request: string,      # Feature request content
    --examples: list,             # Examples to include
    --template_type: string = "prp_base"  # Base template type
] -> record {
    
    print $"🔍 Analyzing project structure for ($target_env)..."
    
    let analysis = (analyze_environment $target_env)
    let patterns = (extract_patterns $analysis $feature_request)
    let context = (gather_context $target_env $feature_request $examples)
    
    let template = (build_dynamic_template $analysis $patterns $context $template_type)
    
    return {
        template: $template,
        analysis: $analysis,
        patterns: $patterns,
        context: $context
    }
}

# Analyze environment structure and capabilities
def analyze_environment [
    env: string                   # Environment to analyze
] -> record {
    
    print $"📊 Analyzing environment: ($env)"
    
    let env_path = if ($env == "multi") { "." } else { $env }
    
    mut analysis = {
        environment: $env,
        structure: {},
        dependencies: {},
        patterns: {},
        conventions: {},
        capabilities: []
    }
    
    # Analyze directory structure
    if ($env_path | path exists) {
        $analysis.structure = (analyze_directory_structure $env_path)
    }
    
    # Analyze dependencies based on environment type
    match $env {
        "python-env" => {
            $analysis.dependencies = (analyze_python_deps $env_path)
            $analysis.patterns = (analyze_python_patterns $env_path)
            $analysis.conventions = (get_python_conventions $env_path)
            $analysis.capabilities = ["fastapi", "async", "sqlalchemy", "pydantic", "pytest"]
        }
        "typescript-env" => {
            $analysis.dependencies = (analyze_typescript_deps $env_path)
            $analysis.patterns = (analyze_typescript_patterns $env_path)
            $analysis.conventions = (get_typescript_conventions $env_path)
            $analysis.capabilities = ["react", "nextjs", "express", "jest", "typescript"]
        }
        "rust-env" => {
            $analysis.dependencies = (analyze_rust_deps $env_path)
            $analysis.patterns = (analyze_rust_patterns $env_path)
            $analysis.conventions = (get_rust_conventions $env_path)
            $analysis.capabilities = ["tokio", "serde", "clap", "cargo"]
        }
        "go-env" => {
            $analysis.dependencies = (analyze_go_deps $env_path)
            $analysis.patterns = (analyze_go_patterns $env_path)
            $analysis.conventions = (get_go_conventions $env_path)
            $analysis.capabilities = ["gin", "cobra", "gorm", "testify"]
        }
        "nushell-env" => {
            $analysis.dependencies = (analyze_nushell_deps $env_path)
            $analysis.patterns = (analyze_nushell_patterns $env_path)
            $analysis.conventions = (get_nushell_conventions $env_path)
            $analysis.capabilities = ["automation", "data-processing", "scripting"]
        }
        "multi" => {
            $analysis = (analyze_multi_environment)
        }
        _ => {
            print $"⚠️  Unknown environment: ($env)"
        }
    }
    
    return $analysis
}

# Analyze directory structure
def analyze_directory_structure [
    path: string                  # Path to analyze
] -> record {
    
    if not ($path | path exists) {
        return {exists: false}
    }
    
    let files = (ls $path | where type == file)
    let dirs = (ls $path | where type == dir)
    
    mut structure = {
        exists: true,
        files: ($files | get name),
        directories: ($dirs | get name),
        config_files: [],
        source_dirs: [],
        test_dirs: []
    }
    
    # Identify key files and directories
    for file in ($files | get name) {
        let filename = ($file | path basename)
        if ($filename in ["package.json", "pyproject.toml", "Cargo.toml", "go.mod", "devbox.json"]) {
            $structure.config_files = ($structure.config_files | append $file)
        }
    }
    
    for dir in ($dirs | get name) {
        let dirname = ($dir | path basename)
        if ($dirname in ["src", "lib", "app"]) {
            $structure.source_dirs = ($structure.source_dirs | append $dir)
        }
        if ($dirname in ["tests", "test", "__tests__", "spec"]) {
            $structure.test_dirs = ($structure.test_dirs | append $dir)
        }
    }
    
    return $structure
}

# Analyze Python dependencies and configuration
def analyze_python_deps [
    path: string                  # Path to Python environment
] -> record {
    
    let pyproject_path = ($path | path join "pyproject.toml")
    let devbox_path = ($path | path join "devbox.json")
    
    mut deps = {
        package_manager: "uv",
        dependencies: [],
        dev_dependencies: [],
        framework: null,
        testing: [],
        linting: []
    }
    
    # Analyze pyproject.toml
    if ($pyproject_path | path exists) {
        let content = (open $pyproject_path)
        
        if ("dependencies" in $content.project?) {
            $deps.dependencies = $content.project.dependencies
        }
        
        if ("optional-dependencies" in $content.project?) and ("dev" in $content.project.optional-dependencies) {
            $deps.dev_dependencies = $content.project.optional-dependencies.dev
        }
        
        # Detect framework
        if ("fastapi" in ($deps.dependencies | str join " ")) {
            $deps.framework = "fastapi"
        } else if ("flask" in ($deps.dependencies | str join " ")) {
            $deps.framework = "flask"
        } else if ("django" in ($deps.dependencies | str join " ")) {
            $deps.framework = "django"
        }
        
        # Detect testing frameworks
        let all_deps = ($deps.dependencies | append $deps.dev_dependencies | str join " ")
        if ("pytest" in $all_deps) {
            $deps.testing = ($deps.testing | append "pytest")
        }
        if ("ruff" in $all_deps) {
            $deps.linting = ($deps.linting | append "ruff")
        }
        if ("mypy" in $all_deps) {
            $deps.linting = ($deps.linting | append "mypy")
        }
    }
    
    return $deps
}

# Analyze TypeScript dependencies and configuration
def analyze_typescript_deps [
    path: string                  # Path to TypeScript environment
] -> record {
    
    let package_path = ($path | path join "package.json")
    
    mut deps = {
        package_manager: "npm",
        dependencies: [],
        dev_dependencies: [],
        framework: null,
        testing: [],
        build_tools: []
    }
    
    if ($package_path | path exists) {
        let content = (open $package_path)
        
        if ("dependencies" in $content) {
            $deps.dependencies = ($content.dependencies | columns)
        }
        
        if ("devDependencies" in $content) {
            $deps.dev_dependencies = ($content.devDependencies | columns)
        }
        
        # Detect package manager
        if (($path | path join "pnpm-lock.yaml") | path exists) {
            $deps.package_manager = "pnpm"
        } else if (($path | path join "yarn.lock") | path exists) {
            $deps.package_manager = "yarn"
        }
        
        # Detect framework
        let all_deps = ($deps.dependencies | append $deps.dev_dependencies)
        if ("next" in $all_deps) {
            $deps.framework = "nextjs"
        } else if ("react" in $all_deps) {
            $deps.framework = "react"
        } else if ("express" in $all_deps) {
            $deps.framework = "express"
        }
        
        # Detect testing and build tools
        if ("jest" in $all_deps) {
            $deps.testing = ($deps.testing | append "jest")
        }
        if ("vitest" in $all_deps) {
            $deps.testing = ($deps.testing | append "vitest")
        }
        if ("eslint" in $all_deps) {
            $deps.build_tools = ($deps.build_tools | append "eslint")
        }
        if ("prettier" in $all_deps) {
            $deps.build_tools = ($deps.build_tools | append "prettier")
        }
    }
    
    return $deps
}

# Extract patterns from feature request
def extract_patterns [
    analysis: record,             # Environment analysis
    feature_request: string       # Feature request content
] -> record {
    
    print "🔍 Extracting patterns from feature request..."
    
    mut patterns = {
        feature_type: "general",
        complexity: "medium",
        integrations: [],
        technologies: [],
        keywords: []
    }
    
    let request_lower = ($feature_request | str downcase)
    
    # Detect feature type
    if ("api" in $request_lower) or ("rest" in $request_lower) or ("endpoint" in $request_lower) {
        $patterns.feature_type = "api"
    } else if ("ui" in $request_lower) or ("component" in $request_lower) or ("frontend" in $request_lower) {
        $patterns.feature_type = "ui"
    } else if ("auth" in $request_lower) or ("login" in $request_lower) or ("user" in $request_lower) {
        $patterns.feature_type = "auth"
    } else if ("database" in $request_lower) or ("data" in $request_lower) or ("model" in $request_lower) {
        $patterns.feature_type = "data"
    } else if ("cli" in $request_lower) or ("command" in $request_lower) or ("tool" in $request_lower) {
        $patterns.feature_type = "cli"
    }
    
    # Detect complexity
    if ("simple" in $request_lower) or ("basic" in $request_lower) {
        $patterns.complexity = "low"
    } else if ("complex" in $request_lower) or ("advanced" in $request_lower) or ("comprehensive" in $request_lower) {
        $patterns.complexity = "high"
    }
    
    # Extract integration requirements
    if ("copilotkit" in $request_lower) {
        $patterns.integrations = ($patterns.integrations | append "copilotkit")
    }
    if ("database" in $request_lower) or ("postgres" in $request_lower) or ("sql" in $request_lower) {
        $patterns.integrations = ($patterns.integrations | append "database")
    }
    if ("react" in $request_lower) {
        $patterns.integrations = ($patterns.integrations | append "react")
    }
    
    # Extract technology keywords
    let tech_keywords = ["fastapi", "nextjs", "tokio", "gin", "typescript", "python", "rust", "go", "nushell"]
    for keyword in $tech_keywords {
        if ($keyword in $request_lower) {
            $patterns.technologies = ($patterns.technologies | append $keyword)
        }
    }
    
    return $patterns
}

# Gather comprehensive context for template generation
def gather_context [
    env: string,                  # Target environment
    feature_request: string,      # Feature request
    examples: list               # Examples to include
] -> record {
    
    print "📚 Gathering context..."
    
    mut context = {
        environment: $env,
        examples: $examples,
        existing_patterns: [],
        documentation: [],
        gotchas: [],
        validation_commands: [],
        integration_points: []
    }
    
    # Gather environment-specific context
    match $env {
        "python-env" => {
            $context.existing_patterns = (get_python_patterns)
            $context.documentation = (get_python_docs)
            $context.gotchas = (get_python_gotchas)
            $context.validation_commands = ["uv run ruff format", "uv run ruff check", "uv run mypy", "uv run pytest"]
        }
        "typescript-env" => {
            $context.existing_patterns = (get_typescript_patterns)
            $context.documentation = (get_typescript_docs)
            $context.gotchas = (get_typescript_gotchas)
            $context.validation_commands = ["npm run format", "npm run lint", "npm run typecheck", "npm run test"]
        }
        _ => {}
    }
    
    # Add dojo-specific context if examples include dojo
    if ("dojo" in $examples) {
        $context = ($context | insert dojo_context (get_dojo_context))
    }
    
    # Add cross-environment integration points
    if ($env == "multi") {
        $context.integration_points = (get_multi_env_integrations)
    }
    
    return $context
}

# Build dynamic template from analysis and context
def build_dynamic_template [
    analysis: record,             # Environment analysis
    patterns: record,             # Extracted patterns
    context: record,              # Gathered context
    template_type: string         # Base template type
] -> string {
    
    print "🏗️  Building dynamic template..."
    
    let header = (generate_template_header $analysis $patterns)
    let context_section = (generate_context_section $analysis $context)
    let implementation_section = (generate_implementation_section $analysis $patterns $context)
    let validation_section = (generate_validation_section $context)
    
    let template = $"($header)

($context_section)

($implementation_section)

($validation_section)"
    
    return $template
}

# Generate template header based on analysis
def generate_template_header [
    analysis: record,             # Environment analysis
    patterns: record              # Extracted patterns
] -> string {
    
    let env_name = match $analysis.environment {
        "python-env" => "Python"
        "typescript-env" => "TypeScript"
        "rust-env" => "Rust"
        "go-env" => "Go"
        "nushell-env" => "Nushell"
        "multi" => "Multi-Environment"
        _ => "Unknown"
    }
    
    let framework = match $analysis.environment {
        "python-env" => ($analysis.dependencies.framework? | default "Python")
        "typescript-env" => ($analysis.dependencies.framework? | default "Node.js")
        _ => ""
    }
    
    let complexity = $patterns.complexity
    let feature_type = $patterns.feature_type
    
    return $"name: \"Dynamic ($env_name) Template - ($feature_type | str title-case) Feature\"
description: |

## Purpose
Dynamically generated template for implementing ($feature_type) features in the ($analysis.environment) environment.
This template is customized based on project analysis and includes relevant patterns, examples, and validation.

## Core Principles
1. **Environment-Specific**: Optimized for ($env_name) development patterns
2. **Context-Aware**: Includes relevant examples and existing code patterns
3. **Validation-Ready**: Comprehensive testing and quality gates
4. **Integration-Focused**: Considers cross-environment dependencies
5. **Modern Standards**: Uses latest 2024 best practices

---"
}

# Generate context section with environment-specific information
def generate_context_section [
    analysis: record,             # Environment analysis
    context: record               # Gathered context
] -> string {
    
    let env_info = (generate_environment_info $analysis)
    let dependencies_info = (generate_dependencies_info $analysis)
    let patterns_info = (generate_patterns_info $context)
    let examples_info = (generate_examples_info $context)
    
    return $"## All Needed Context

### Target Environment
```yaml
Environment: ($analysis.environment)
($env_info)
```

### Dependencies and Tools
($dependencies_info)

### Existing Patterns
($patterns_info)

### Examples and References
($examples_info)"
}

# Helper functions for generating specific sections
def generate_environment_info [analysis: record] -> string {
    match $analysis.environment {
        "python-env" => {
            return $"Python_Version: 3.12+ (from devbox.json)
Package_Manager: uv (exclusively)
Framework: ($analysis.dependencies.framework? | default 'FastAPI')
Testing: ($analysis.dependencies.testing | str join ', ')
Linting: ($analysis.dependencies.linting | str join ', ')"
        }
        "typescript-env" => {
            return $"Node_Version: 20+ (from devbox.json)
Package_Manager: ($analysis.dependencies.package_manager)
Framework: ($analysis.dependencies.framework? | default 'Node.js')
Testing: ($analysis.dependencies.testing | str join ', ')
Build_Tools: ($analysis.dependencies.build_tools | str join ', ')"
        }
        _ => "Auto-detected configuration"
    }
}

def generate_dependencies_info [analysis: record] -> string {
    if (($analysis.dependencies.dependencies? | default [] | length) > 0) {
        return $"```yaml
Production: ($analysis.dependencies.dependencies | str join ', ')
Development: ($analysis.dependencies.dev_dependencies | str join ', ')
```"
    } else {
        return "Dependencies will be auto-detected from project configuration."
    }
}

def generate_patterns_info [context: record] -> string {
    if (($context.existing_patterns | length) > 0) {
        return ($context.existing_patterns | str join "\n- ")
    } else {
        return "Patterns will be extracted from existing codebase during implementation."
    }
}

def generate_examples_info [context: record] -> string {
    if (($context.examples | length) > 0) {
        return ($context.examples | each { |ex| $"- ($ex): Relevant patterns and implementation examples" } | str join "\n")
    } else {
        return "No specific examples requested - will use environment defaults."
    }
}

# Generate implementation section
def generate_implementation_section [
    analysis: record,             # Environment analysis
    patterns: record,             # Extracted patterns
    context: record               # Gathered context
] -> string {
    
    let setup_commands = (generate_setup_commands $analysis)
    let task_list = (generate_task_list $analysis $patterns)
    let code_examples = (generate_code_examples $analysis $patterns)
    
    return $"## Implementation Blueprint

### Environment Setup
($setup_commands)

### Task List
($task_list)

### Code Examples and Patterns
($code_examples)"
}

def generate_setup_commands [analysis: record] -> string {
    match $analysis.environment {
        "python-env" => {
            return $"```bash
# Activate Python environment
cd python-env && devbox shell

# Verify environment
uv --version
python --version

# Install dependencies (will be determined during implementation)
uv add <required-packages>
```"
        }
        "typescript-env" => {
            return $"```bash
# Activate TypeScript environment
cd typescript-env && devbox shell

# Verify environment
node --version
npm --version

# Install dependencies (will be determined during implementation)
npm install <required-packages>
```"
        }
        _ => {
            return $"```bash
# Activate ($analysis.environment)
cd ($analysis.environment) && devbox shell

# Environment-specific setup will be determined during implementation
```"
        }
    }
}

def generate_task_list [analysis: record, patterns: record] -> string {
    match $patterns.feature_type {
        "api" => {
            return generate_api_tasks $analysis
        }
        "ui" => {
            return generate_ui_tasks $analysis
        }
        "auth" => {
            return generate_auth_tasks $analysis
        }
        _ => {
            return generate_general_tasks $analysis
        }
    }
}

def generate_api_tasks [analysis: record] -> string {
    match $analysis.environment {
        "python-env" => {
            return $"```yaml
Task 1: API Design
  - Define endpoints and data models
  - Create Pydantic schemas
  - Plan database integration

Task 2: Core Implementation
  - Implement FastAPI routes
  - Add dependency injection
  - Create service layer

Task 3: Database Integration
  - Set up SQLAlchemy models
  - Create database migrations
  - Implement CRUD operations

Task 4: Testing
  - Write unit tests with pytest
  - Add integration tests
  - Test API endpoints

Task 5: Validation and Deployment
  - Run linting and type checking
  - Validate API documentation
  - Prepare for deployment
```"
        }
        _ => {
            return $"```yaml
Task 1: API Design and Planning
Task 2: Core Implementation
Task 3: Integration Points
Task 4: Testing and Validation
Task 5: Documentation and Deployment
```"
        }
    }
}

def generate_ui_tasks [analysis: record] -> string {
    return $"```yaml
Task 1: Component Design
  - Plan component structure
  - Define props and state
  - Create wireframes

Task 2: Implementation
  - Build core components
  - Add styling and theming
  - Implement interactions

Task 3: Integration
  - Connect to backend APIs
  - Add state management
  - Handle error states

Task 4: Testing
  - Unit test components
  - Integration testing
  - Accessibility testing

Task 5: Polish and Deploy
  - Performance optimization
  - Cross-browser testing
  - Production deployment
```"
}

def generate_auth_tasks [analysis: record] -> string {
    return $"```yaml
Task 1: Authentication Design
  - Choose auth strategy
  - Plan user flows
  - Design security model

Task 2: Backend Implementation
  - User registration/login
  - Token management
  - Password security

Task 3: Frontend Integration
  - Login/signup forms
  - Protected routes
  - User state management

Task 4: Security Testing
  - Penetration testing
  - Input validation
  - Session management

Task 5: Production Security
  - Security headers
  - Rate limiting
  - Monitoring setup
```"
}

def generate_general_tasks [analysis: record] -> string {
    return $"```yaml
Task 1: Requirements Analysis
  - Understand feature scope
  - Identify dependencies
  - Plan implementation approach

Task 2: Core Development
  - Implement main functionality
  - Add error handling
  - Create documentation

Task 3: Integration
  - Connect with existing systems
  - Test compatibility
  - Handle edge cases

Task 4: Quality Assurance
  - Comprehensive testing
  - Code review
  - Performance validation

Task 5: Deployment
  - Production preparation
  - Monitoring setup
  - Documentation updates
```"
}

def generate_code_examples [analysis: record, patterns: record] -> string {
    # This would include relevant code snippets based on the analysis
    return $"```
Code examples will be dynamically generated based on:
- Existing codebase patterns
- Target environment conventions
- Feature type requirements
- Integration needs
```"
}

# Generate validation section
def generate_validation_section [context: record] -> string {
    let commands = ($context.validation_commands | each { |cmd| $"($cmd)" } | str join "\n")
    
    return $"## Validation Loop

### Environment-Specific Validation
```bash
($commands)
```

### Integration Testing
- Cross-environment compatibility checks
- Performance validation
- Security scanning

### Success Criteria
- All tests pass
- Code quality gates met
- Integration points verified
- Documentation complete"
}

# Helper functions for getting environment-specific information
def get_python_patterns [] -> list {
    return [
        "Use async/await for all FastAPI endpoints",
        "Pydantic v2 for data validation",
        "SQLAlchemy 2.0 async patterns",
        "Dependency injection with FastAPI",
        "Error handling with custom exceptions"
    ]
}

def get_python_docs [] -> list {
    return [
        "https://fastapi.tiangolo.com/",
        "https://docs.pydantic.dev/",
        "https://docs.sqlalchemy.org/en/20/"
    ]
}

def get_python_gotchas [] -> list {
    return [
        "Use uv exclusively, not pip/poetry",
        "Always use async def for FastAPI endpoints",
        "PyJWT instead of python-jose (deprecated)",
        "SQLAlchemy 2.0 syntax, not 1.x patterns"
    ]
}

def get_typescript_patterns [] -> list {
    return [
        "Strict TypeScript mode enabled",
        "No any types allowed",
        "React functional components with hooks",
        "Next.js app router patterns",
        "Proper error boundaries"
    ]
}

def get_typescript_docs [] -> list {
    return [
        "https://nextjs.org/docs",
        "https://react.dev/",
        "https://www.typescriptlang.org/docs/"
    ]
}

def get_typescript_gotchas [] -> list {
    return [
        "Never use any type",
        "Prefer interfaces over types",
        "Use Result pattern for error handling",
        "Always handle loading and error states"
    ]
}

def get_dojo_context [] -> record {
    return {
        framework: "CopilotKit",
        patterns: [
            "Interactive agent demos",
            "Generative UI components",
            "Real-time collaboration",
            "Tool-based interactions"
        ],
        integration_points: [
            "Agent configuration",
            "UI component structure",
            "State management patterns",
            "Event handling"
        ]
    }
}

def get_multi_env_integrations [] -> list {
    return [
        "API communication between environments",
        "Shared data models and schemas",
        "Cross-environment testing",
        "Unified monitoring and logging",
        "DevOps and deployment coordination"
    ]
}

def analyze_rust_deps [path: string] -> record { return {} }
def analyze_go_deps [path: string] -> record { return {} }
def analyze_nushell_deps [path: string] -> record { return {} }

def analyze_python_patterns [path: string] -> record { return {} }
def analyze_typescript_patterns [path: string] -> record { return {} }
def analyze_rust_patterns [path: string] -> record { return {} }
def analyze_go_patterns [path: string] -> record { return {} }
def analyze_nushell_patterns [path: string] -> record { return {} }

def get_python_conventions [path: string] -> record { return {} }
def get_typescript_conventions [path: string] -> record { return {} }
def get_rust_conventions [path: string] -> record { return {} }
def get_go_conventions [path: string] -> record { return {} }
def get_nushell_conventions [path: string] -> record { return {} }

def analyze_multi_environment [] -> record { return {} }

# Export main functions
export def main [] {
    print "Dynamic Template Generation Engine"
    print "Usage: Use 'generate dynamic template' to create custom templates"
}