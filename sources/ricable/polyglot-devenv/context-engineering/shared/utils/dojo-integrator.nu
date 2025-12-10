#!/usr/bin/env nu
# Dojo Integration System for Context Engineering
# Parses dojo app structure and extracts reusable patterns for PRPs

# Parse dojo app structure and extract patterns
def "parse dojo patterns" [] -> record {
    
    print "🥋 Parsing dojo app structure..."
    
    let dojo_path = "context-engineering/examples/dojo"
    
    if not ($dojo_path | path exists) {
        print $"❌ Dojo app not found at ($dojo_path)"
        return {}
    }
    
    let structure = (analyze_dojo_structure $dojo_path)
    let features = (extract_dojo_features $dojo_path)
    let patterns = (extract_code_patterns $dojo_path)
    let config = (extract_configuration_patterns $dojo_path)
    let components = (extract_component_patterns $dojo_path)
    
    return {
        structure: $structure,
        features: $features,
        patterns: $patterns,
        configuration: $config,
        components: $components,
        integration_guide: (generate_integration_guide $features $patterns)
    }
}

# Analyze dojo directory structure
def analyze_dojo_structure [path: string] -> record {
    
    let structure = (ls -la $path | where type == dir)
    let files = (ls -la $path | where type == file)
    
    mut analysis = {
        root_files: [],
        source_structure: {},
        feature_structure: {},
        config_files: [],
        component_dirs: []
    }
    
    # Analyze root files
    for file in $files {
        let filename = ($file.name | path basename)
        if ($filename in ["package.json", "next.config.ts", "tailwind.config.ts", "tsconfig.json"]) {
            $analysis.config_files = ($analysis.config_files | append $file.name)
        }
        $analysis.root_files = ($analysis.root_files | append $filename)
    }
    
    # Analyze source structure if src directory exists
    let src_path = ($path | path join "src")
    if ($src_path | path exists) {
        $analysis.source_structure = (analyze_src_structure $src_path)
    }
    
    return $analysis
}

# Analyze src directory structure
def analyze_src_structure [src_path: string] -> record {
    
    mut src_analysis = {
        app_structure: {},
        components: [],
        types: [],
        utils: [],
        config_files: []
    }
    
    # Analyze app directory (Next.js app router)
    let app_path = ($src_path | path join "app")
    if ($app_path | path exists) {
        $src_analysis.app_structure = (analyze_app_structure $app_path)
    }
    
    # Analyze components directory
    let components_path = ($src_path | path join "components")
    if ($components_path | path exists) {
        $src_analysis.components = (analyze_components_structure $components_path)
    }
    
    # Analyze types directory
    let types_path = ($src_path | path join "types")
    if ($types_path | path exists) {
        let type_files = (ls $types_path | where type == file | get name)
        $src_analysis.types = $type_files
    }
    
    # Analyze config files
    for file in ["config.ts", "menu.ts"] {
        let file_path = ($src_path | path join $file)
        if ($file_path | path exists) {
            $src_analysis.config_files = ($src_analysis.config_files | append $file_path)
        }
    }
    
    return $src_analysis
}

# Analyze Next.js app structure
def analyze_app_structure [app_path: string] -> record {
    
    mut app_analysis = {
        routes: [],
        api_routes: [],
        layouts: [],
        features: {},
        dynamic_routes: []
    }
    
    # Find all directories (routes)
    let dirs = (ls $app_path | where type == dir)
    
    for dir in $dirs {
        let dir_name = ($dir.name | path basename)
        
        if ($dir_name == "api") {
            # Analyze API routes
            $app_analysis.api_routes = (analyze_api_routes ($dir.name))
        } else if ($dir_name | str starts-with "[") {
            # Dynamic route
            $app_analysis.dynamic_routes = ($app_analysis.dynamic_routes | append {
                path: $dir.name,
                pattern: $dir_name,
                features: (analyze_feature_directory ($dir.name))
            })
        } else {
            # Regular route
            $app_analysis.routes = ($app_analysis.routes | append $dir_name)
        }
    }
    
    # Analyze [integrationId] dynamic route specifically
    let integration_path = ($app_path | path join "[integrationId]")
    if ($integration_path | path exists) {
        $app_analysis.features = (analyze_integration_features $integration_path)
    }
    
    return $app_analysis
}

# Analyze integration features (CopilotKit features)
def analyze_integration_features [integration_path: string] -> record {
    
    mut features = {}
    
    let feature_path = ($integration_path | path join "feature")
    if ($feature_path | path exists) {
        let feature_dirs = (ls $feature_path | where type == dir)
        
        for feature_dir in $feature_dirs {
            let feature_name = ($feature_dir.name | path basename)
            let feature_files = (ls ($feature_dir.name) | where type == file | get name)
            
            $features = ($features | insert $feature_name {
                files: $feature_files,
                has_readme: (($feature_files | any { |f| ($f | str ends-with "README.mdx") }) | default false),
                has_page: (($feature_files | any { |f| ($f | str ends-with "page.tsx") }) | default false),
                has_styles: (($feature_files | any { |f| ($f | str ends-with ".css") }) | default false)
            })
        }
    }
    
    return $features
}

# Extract dojo features and their patterns
def extract_dojo_features [path: string] -> record {
    
    print "🔍 Extracting dojo features..."
    
    let config_path = ($path | path join "src" | path join "config.ts")
    
    mut features = {
        defined_features: [],
        feature_patterns: {},
        copilotkit_integrations: []
    }
    
    # Read feature configuration
    if ($config_path | path exists) {
        let config_content = (open $config_path)
        $features.defined_features = (extract_feature_configs $config_content)
    }
    
    # Analyze each feature directory
    let features_path = ($path | path join "src" | path join "app" | path join "[integrationId]" | path join "feature")
    if ($features_path | path exists) {
        let feature_dirs = (ls $features_path | where type == dir)
        
        for feature_dir in $feature_dirs {
            let feature_name = ($feature_dir.name | path basename)
            let patterns = (analyze_feature_patterns ($feature_dir.name))
            $features.feature_patterns = ($features.feature_patterns | insert $feature_name $patterns)
        }
    }
    
    return $features
}

# Extract feature configurations from config.ts
def extract_feature_configs [config_content: string] -> list {
    
    mut features = []
    
    # Parse the featureConfig array (simplified parsing)
    let lines = ($config_content | split row "\n")
    mut in_feature_config = false
    mut current_feature = {}
    
    for line in $lines {
        let trimmed = ($line | str trim)
        
        if ($trimmed | str contains "createFeatureConfig") {
            $in_feature_config = true
            $current_feature = {}
        } else if ($in_feature_config and ($trimmed | str contains "id:")) {
            let id = ($trimmed | str replace 'id: "' '' | str replace '",' '')
            $current_feature = ($current_feature | insert id $id)
        } else if ($in_feature_config and ($trimmed | str contains "name:")) {
            let name = ($trimmed | str replace 'name: "' '' | str replace '",' '')
            $current_feature = ($current_feature | insert name $name)
        } else if ($in_feature_config and ($trimmed | str contains "description:")) {
            let desc = ($trimmed | str replace 'description: "' '' | str replace '",' '')
            $current_feature = ($current_feature | insert description $desc)
        } else if ($in_feature_config and ($trimmed | str contains "tags:")) {
            # Extract tags array (simplified)
            let tags_line = ($trimmed | str replace 'tags: [' '' | str replace '],' '')
            let tags = ($tags_line | split row ',' | each { |tag| $tag | str trim | str replace '"' '' })
            $current_feature = ($current_feature | insert tags $tags)
        } else if ($in_feature_config and ($trimmed | str contains "})")) {
            $in_feature_config = false
            if (($current_feature | columns | length) > 0) {
                $features = ($features | append $current_feature)
            }
        }
    }
    
    return $features
}

# Analyze patterns in a specific feature directory
def analyze_feature_patterns [feature_path: string] -> record {
    
    let files = (ls $feature_path | where type == file)
    
    mut patterns = {
        components: [],
        hooks: [],
        styles: [],
        documentation: [],
        copilotkit_usage: [],
        react_patterns: []
    }
    
    for file in $files {
        let filename = ($file.name | path basename)
        let extension = ($filename | path parse | get extension)
        
        match $extension {
            "tsx" => {
                let tsx_patterns = (analyze_tsx_file ($file.name))
                $patterns.components = ($patterns.components | append $tsx_patterns.components)
                $patterns.hooks = ($patterns.hooks | append $tsx_patterns.hooks)
                $patterns.copilotkit_usage = ($patterns.copilotkit_usage | append $tsx_patterns.copilotkit)
                $patterns.react_patterns = ($patterns.react_patterns | append $tsx_patterns.react_patterns)
            }
            "css" => {
                $patterns.styles = ($patterns.styles | append (analyze_css_file ($file.name)))
            }
            "mdx" => {
                $patterns.documentation = ($patterns.documentation | append (analyze_mdx_file ($file.name)))
            }
            _ => {}
        }
    }
    
    return $patterns
}

# Analyze TypeScript React component file
def analyze_tsx_file [file_path: string] -> record {
    
    let content = (open $file_path)
    
    mut analysis = {
        components: [],
        hooks: [],
        copilotkit: [],
        react_patterns: [],
        imports: []
    }
    
    let lines = ($content | split row "\n")
    
    for line in $lines {
        let trimmed = ($line | str trim)
        
        # Extract imports
        if ($trimmed | str starts-with "import") {
            $analysis.imports = ($analysis.imports | append $trimmed)
            
            # Check for CopilotKit imports
            if ($trimmed | str contains "copilotkit") or ($trimmed | str contains "CopilotKit") {
                let copilot_import = (extract_copilotkit_import $trimmed)
                if ($copilot_import != null) {
                    $analysis.copilotkit = ($analysis.copilotkit | append $copilot_import)
                }
            }
        }
        
        # Extract component definitions
        if ($trimmed | str contains "export default function") or ($trimmed | str contains "function") {
            let component = (extract_component_name $trimmed)
            if ($component != null) {
                $analysis.components = ($analysis.components | append $component)
            }
        }
        
        # Extract React hooks usage
        if ($trimmed | str contains "useState") or ($trimmed | str contains "useEffect") or ($trimmed | str contains "use") {
            let hook = (extract_hook_usage $trimmed)
            if ($hook != null) {
                $analysis.hooks = ($analysis.hooks | append $hook)
            }
        }
        
        # Extract React patterns
        if ($trimmed | str contains "onClick") or ($trimmed | str contains "onChange") or ($trimmed | str contains "onSubmit") {
            $analysis.react_patterns = ($analysis.react_patterns | append "event_handlers")
        }
        
        if ($trimmed | str contains "className") {
            $analysis.react_patterns = ($analysis.react_patterns | append "tailwind_styling")
        }
    }
    
    return $analysis
}

# Extract code patterns from dojo
def extract_code_patterns [path: string] -> record {
    
    print "📋 Extracting code patterns..."
    
    mut patterns = {
        typescript_patterns: [],
        react_patterns: [],
        nextjs_patterns: [],
        copilotkit_patterns: [],
        styling_patterns: [],
        testing_patterns: []
    }
    
    # Analyze TypeScript configuration
    let tsconfig_path = ($path | path join "tsconfig.json")
    if ($tsconfig_path | path exists) {
        let tsconfig = (open $tsconfig_path)
        $patterns.typescript_patterns = (extract_typescript_config_patterns $tsconfig)
    }
    
    # Analyze Next.js configuration
    let nextconfig_path = ($path | path join "next.config.ts")
    if ($nextconfig_path | path exists) {
        let nextconfig = (open $nextconfig_path)
        $patterns.nextjs_patterns = (extract_nextjs_config_patterns $nextconfig)
    }
    
    # Analyze Tailwind configuration
    let tailwind_path = ($path | path join "tailwind.config.ts")
    if ($tailwind_path | path exists) {
        let tailwind = (open $tailwind_path)
        $patterns.styling_patterns = (extract_tailwind_patterns $tailwind)
    }
    
    # Analyze package.json for dependencies and scripts
    let package_path = ($path | path join "package.json")
    if ($package_path | path exists) {
        let package_json = (open $package_path)
        let dep_patterns = (extract_dependency_patterns $package_json)
        $patterns.react_patterns = ($patterns.react_patterns | append $dep_patterns.react)
        $patterns.copilotkit_patterns = ($patterns.copilotkit_patterns | append $dep_patterns.copilotkit)
    }
    
    return $patterns
}

# Extract configuration patterns
def extract_configuration_patterns [path: string] -> record {
    
    print "⚙️  Extracting configuration patterns..."
    
    mut config = {
        build_config: {},
        dependencies: {},
        scripts: {},
        linting: {},
        formatting: {}
    }
    
    # Extract package.json configuration
    let package_path = ($path | path join "package.json")
    if ($package_path | path exists) {
        let package_json = (open $package_path)
        
        if ("scripts" in $package_json) {
            $config.scripts = $package_json.scripts
        }
        
        if ("dependencies" in $package_json) {
            $config.dependencies = ($config.dependencies | insert prod $package_json.dependencies)
        }
        
        if ("devDependencies" in $package_json) {
            $config.dependencies = ($config.dependencies | insert dev $package_json.devDependencies)
        }
    }
    
    # Extract ESLint configuration
    let eslint_path = ($path | path join "eslint.config.mjs")
    if ($eslint_path | path exists) {
        $config.linting = {eslint_config: $eslint_path}
    }
    
    # Extract PostCSS configuration
    let postcss_path = ($path | path join "postcss.config.mjs")
    if ($postcss_path | path exists) {
        $config.formatting = ($config.formatting | insert postcss $postcss_path)
    }
    
    return $config
}

# Extract component patterns from dojo
def extract_component_patterns [path: string] -> record {
    
    print "🧩 Extracting component patterns..."
    
    let components_path = ($path | path join "src" | path join "components")
    
    mut component_patterns = {
        ui_components: [],
        layout_components: [],
        feature_components: [],
        reusable_patterns: []
    }
    
    if ($components_path | path exists) {
        let component_dirs = (ls $components_path | where type == dir)
        
        for comp_dir in $component_dirs {
            let dir_name = ($comp_dir.name | path basename)
            let files = (ls ($comp_dir.name) | where type == file | get name)
            
            match $dir_name {
                "ui" => {
                    $component_patterns.ui_components = $files
                }
                "layout" => {
                    $component_patterns.layout_components = $files
                }
                _ => {
                    $component_patterns.feature_components = ($component_patterns.feature_components | append {
                        name: $dir_name,
                        files: $files
                    })
                }
            }
        }
    }
    
    return $component_patterns
}

# Generate integration guide for using dojo patterns in PRPs
def generate_integration_guide [features: record, patterns: record] -> record {
    
    print "📖 Generating integration guide..."
    
    mut guide = {
        feature_integration: {},
        pattern_usage: {},
        copilotkit_integration: {},
        component_reuse: {},
        configuration_templates: {}
    }
    
    # Feature integration guidance
    for feature_name in ($features.feature_patterns | columns) {
        let feature_data = ($features.feature_patterns | get $feature_name)
        
        $guide.feature_integration = ($guide.feature_integration | insert $feature_name {
            description: $"Integrate ($feature_name) patterns from dojo",
            components: ($feature_data.components | uniq),
            copilotkit_usage: ($feature_data.copilotkit_usage | uniq),
            applicable_when: (generate_applicability_rules $feature_name $feature_data)
        })
    }
    
    # Pattern usage guidance
    $guide.pattern_usage = {
        react_patterns: ($patterns.react_patterns | uniq),
        typescript_patterns: ($patterns.typescript_patterns | uniq),
        styling_patterns: ($patterns.styling_patterns | uniq),
        nextjs_patterns: ($patterns.nextjs_patterns | uniq)
    }
    
    # CopilotKit integration templates
    $guide.copilotkit_integration = {
        basic_setup: "Include CopilotKit provider and basic configuration",
        agent_integration: "Add agent configuration and tool definitions",
        ui_components: "Use CopilotKit UI components for chat and interaction",
        state_management: "Implement shared state patterns for agent-UI communication"
    }
    
    return $guide
}

# Generate rules for when to apply specific dojo patterns
def generate_applicability_rules [feature_name: string, feature_data: record] -> list {
    
    mut rules = []
    
    match $feature_name {
        "agentic_chat" => {
            $rules = [
                "When feature involves chat or conversation interfaces",
                "When implementing agent communication patterns",
                "When building streaming response interfaces"
            ]
        }
        "human_in_the_loop" => {
            $rules = [
                "When feature requires user approval or intervention",
                "When implementing collaborative workflows",
                "When building interactive planning interfaces"
            ]
        }
        "agentic_generative_ui" => {
            $rules = [
                "When feature involves dynamic UI generation",
                "When implementing long-running agent tasks",
                "When building adaptive user interfaces"
            ]
        }
        "shared_state" => {
            $rules = [
                "When feature requires real-time collaboration",
                "When implementing shared data structures",
                "When building multi-user interfaces"
            ]
        }
        _ => {
            $rules = ["Apply when feature requirements match dojo patterns"]
        }
    }
    
    return $rules
}

# Helper functions for pattern extraction
def extract_copilotkit_import [import_line: string] -> string {
    if ($import_line | str contains "copilotkit") {
        return $import_line
    }
    return null
}

def extract_component_name [line: string] -> string {
    # Extract component name from function definition
    if ($line | str contains "function") {
        let parts = ($line | split row " ")
        for part in $parts {
            if ($part | str contains "(") {
                return ($part | str replace "(" "")
            }
        }
    }
    return null
}

def extract_hook_usage [line: string] -> string {
    if ($line | str contains "use") {
        return ($line | str trim)
    }
    return null
}

def extract_typescript_config_patterns [tsconfig: record] -> list {
    mut patterns = []
    
    if ("compilerOptions" in $tsconfig) {
        let options = $tsconfig.compilerOptions
        if ("strict" in $options) and ($options.strict == true) {
            $patterns = ($patterns | append "strict_mode_enabled")
        }
        if ("target" in $options) {
            $patterns = ($patterns | append $"target_($options.target)")
        }
    }
    
    return $patterns
}

def extract_nextjs_config_patterns [nextconfig: string] -> list {
    mut patterns = []
    
    if ($nextconfig | str contains "experimental") {
        $patterns = ($patterns | append "experimental_features")
    }
    if ($nextconfig | str contains "typescript") {
        $patterns = ($patterns | append "typescript_integration")
    }
    
    return $patterns
}

def extract_tailwind_patterns [tailwind: string] -> list {
    mut patterns = []
    
    if ($tailwind | str contains "darkMode") {
        $patterns = ($patterns | append "dark_mode_support")
    }
    if ($tailwind | str contains "plugins") {
        $patterns = ($patterns | append "plugin_usage")
    }
    
    return $patterns
}

def extract_dependency_patterns [package_json: record] -> record {
    mut patterns = {react: [], copilotkit: []}
    
    let all_deps = if ("dependencies" in $package_json) { 
        ($package_json.dependencies | columns) 
    } else { 
        [] 
    }
    
    for dep in $all_deps {
        if ($dep | str contains "react") {
            $patterns.react = ($patterns.react | append $dep)
        }
        if ($dep | str contains "copilotkit") {
            $patterns.copilotkit = ($patterns.copilotkit | append $dep)
        }
    }
    
    return $patterns
}

def analyze_api_routes [api_path: string] -> list {
    let routes = (ls $api_path -r | where type == file | get name)
    return $routes
}

def analyze_feature_directory [dir_path: string] -> list {
    let files = (ls $dir_path | where type == file | get name)
    return $files
}

def analyze_components_structure [components_path: string] -> list {
    let components = (ls $components_path -r | where type == file | get name)
    return $components
}

def analyze_css_file [file_path: string] -> record {
    return {file: $file_path, patterns: ["css_modules", "responsive_design"]}
}

def analyze_mdx_file [file_path: string] -> record {
    return {file: $file_path, type: "documentation"}
}

# Export main function
export def main [] {
    print "Dojo Integration System"
    print "Usage: Use 'parse dojo patterns' to extract reusable patterns"
    
    # Example usage
    let patterns = (parse dojo patterns)
    print $"Found ($patterns.features.defined_features | length) features in dojo"
}