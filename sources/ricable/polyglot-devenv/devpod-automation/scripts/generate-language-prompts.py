#!/usr/bin/env python3
"""
Language-Specific Prompt Generator for Agentic Evaluation Framework
Creates tailored evaluation prompts for each programming language ecosystem
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import argparse

class LanguagePromptGenerator:
    def __init__(self, output_dir: str = "/workspace/agentic-eval/prompts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Language-specific configurations
        self.language_configs = {
            "python": {
                "frameworks": ["FastAPI", "Flask", "Django", "Streamlit"],
                "testing": ["pytest", "unittest", "doctest"],
                "tools": ["ruff", "mypy", "black"],
                "patterns": ["async/await", "type hints", "context managers", "decorators"],
                "package_manager": "uv",
                "file_extension": ".py"
            },
            "typescript": {
                "frameworks": ["React", "Vue", "Express", "Next.js"],
                "testing": ["Jest", "Vitest", "Cypress"],
                "tools": ["ESLint", "Prettier", "TypeScript compiler"],
                "patterns": ["interfaces", "generics", "async/await", "modules"],
                "package_manager": "npm",
                "file_extension": ".ts"
            },
            "rust": {
                "frameworks": ["Tokio", "Actix", "Axum", "Warp"],
                "testing": ["cargo test", "proptest", "criterion"],
                "tools": ["rustfmt", "clippy", "cargo"],
                "patterns": ["ownership", "Result<T,E>", "Option<T>", "traits"],
                "package_manager": "cargo",
                "file_extension": ".rs"
            },
            "go": {
                "frameworks": ["Gin", "Echo", "Fiber", "Chi"],
                "testing": ["go test", "testify", "Ginkgo"],
                "tools": ["gofmt", "golangci-lint", "go mod"],
                "patterns": ["interfaces", "channels", "goroutines", "error handling"],
                "package_manager": "go mod",
                "file_extension": ".go"
            },
            "nushell": {
                "frameworks": ["Nushell", "Custom commands", "Modules"],
                "testing": ["nu test", "custom test harness"],
                "tools": ["nu format", "nu check"],
                "patterns": ["pipelines", "structured data", "custom commands", "hooks"],
                "package_manager": "nu",
                "file_extension": ".nu"
            }
        }
        
        # Complexity level descriptions
        self.complexity_levels = {
            1: {"name": "Beginner", "description": "Basic syntax and simple operations"},
            2: {"name": "Intermediate", "description": "Common patterns and moderate complexity"},
            3: {"name": "Advanced", "description": "Complex logic and framework integration"},
            4: {"name": "Expert", "description": "Performance optimization and advanced patterns"},
            5: {"name": "Master", "description": "Architectural decisions and complex systems"}
        }
        
        # Evaluation categories
        self.categories = {
            "ui-components": "User Interface Components",
            "apis": "REST API Development", 
            "cli-tools": "Command Line Tools",
            "web-apps": "Web Applications",
            "data-processing": "Data Processing and Analysis",
            "refactoring": "Code Refactoring and Optimization"
        }

    def generate_all_prompts(self):
        """Generate all language-specific prompts"""
        print("🚀 Generating language-specific evaluation prompts...")
        
        for language in self.language_configs.keys():
            print(f"📝 Generating prompts for {language}...")
            self.generate_language_prompts(language)
        
        print("✅ All language-specific prompts generated!")

    def generate_language_prompts(self, language: str):
        """Generate prompts for a specific language"""
        config = self.language_configs[language]
        lang_dir = self.output_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        for category in self.categories.keys():
            category_dir = lang_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            for level in range(1, 6):
                prompt = self.create_prompt(language, category, level, config)
                filename = f"level_{level}_{category}.md"
                
                with open(category_dir / filename, 'w') as f:
                    f.write(prompt)
                
                print(f"  ✅ Created {language}/{category}/level_{level}_{category}.md")

    def create_prompt(self, language: str, category: str, level: int, config: Dict) -> str:
        """Create a specific prompt for language, category, and complexity level"""
        complexity = self.complexity_levels[level]
        category_name = self.categories[category]
        
        # Language-specific prompt generation based on category
        if category == "ui-components":
            return self.create_ui_component_prompt(language, level, config, complexity)
        elif category == "apis":
            return self.create_api_prompt(language, level, config, complexity)
        elif category == "cli-tools":
            return self.create_cli_prompt(language, level, config, complexity)
        elif category == "web-apps":
            return self.create_webapp_prompt(language, level, config, complexity)
        elif category == "data-processing":
            return self.create_data_processing_prompt(language, level, config, complexity)
        elif category == "refactoring":
            return self.create_refactoring_prompt(language, level, config, complexity)
        
        return self.create_generic_prompt(language, category, level, config, complexity)

    def create_ui_component_prompt(self, language: str, level: int, config: Dict, complexity: Dict) -> str:
        """Create UI component prompt specific to language"""
        frameworks = config.get("frameworks", [])
        testing = config.get("testing", [])
        tools = config.get("tools", [])
        
        if language == "python":
            component_types = ["Streamlit widget", "Flask template", "FastAPI response model"]
            tech_stack = "Streamlit/Flask/FastAPI"
        elif language == "typescript":
            component_types = ["React component", "Vue component", "Custom HTML element"]
            tech_stack = "React/Vue/Vanilla TS"
        elif language == "rust":
            component_types = ["CLI interface", "Web component (WASM)", "Terminal UI"]
            tech_stack = "WASM/Yew/Ratatui"
        elif language == "go":
            component_types = ["HTML template", "CLI interface", "Web component"]
            tech_stack = "html/template/CLI"
        else:  # nushell
            component_types = ["Custom command", "Table formatter", "Data visualizer"]
            tech_stack = "Nushell commands"
        
        # Complexity-specific requirements
        complexity_requirements = {
            1: "Simple display component with basic styling",
            2: "Interactive component with user input handling",
            3: "Component with state management and validation",
            4: "Complex component with performance optimization",
            5: "Reusable component library with advanced patterns"
        }
        
        return f"""# {language.title()} UI Component Evaluation - Level {level} ({complexity['name']})

## Task Description
Create a {component_types[min(level-1, len(component_types)-1)]} in {language.title()} that demonstrates {complexity['description'].lower()}.

**Component Requirements:**
{complexity_requirements[level]}

**Technical Stack:**
- Language: {language.title()}
- Framework: {tech_stack}
- Testing: {testing[0] if testing else 'Manual testing'}
- Tools: {', '.join(tools[:2])}

**Specific Requirements:**
{'- Responsive design and accessibility' if level >= 2 else '- Basic functionality'}
{'- Error handling and validation' if level >= 3 else ''}
{'- Performance optimization' if level >= 4 else ''}
{'- Comprehensive documentation and examples' if level >= 5 else ''}

## Language-Specific Patterns
Apply these {language.title()} patterns where appropriate:
- {config['patterns'][0]}
- {config['patterns'][1] if len(config['patterns']) > 1 else 'Standard conventions'}

## Evaluation Criteria
1. **Visual Fidelity** (25%): Component appearance and user experience
2. **Code Quality** (25%): {language.title()}-specific best practices and patterns
3. **Functionality** (25%): Feature completeness and reliability
4. **Performance** (15%): Optimization and efficiency
5. **Maintainability** (10%): Documentation and code organization

## Expected Deliverables
- Component implementation ({config['file_extension']} files)
- Test cases using {testing[0] if testing else 'manual testing'}
- Usage documentation
{'- Performance benchmarks' if level >= 4 else ''}
{'- Accessibility audit' if level >= 3 else ''}

## Complexity Level: {level}/5 ({complexity['name']})
**Focus**: {complexity['description']}
"""

    def create_api_prompt(self, language: str, level: int, config: Dict, complexity: Dict) -> str:
        """Create API development prompt specific to language"""
        frameworks = config.get("frameworks", [])
        
        if language == "python":
            api_framework = "FastAPI" if level >= 3 else "Flask"
            db_tech = "SQLAlchemy + PostgreSQL"
        elif language == "typescript":
            api_framework = "Express.js" if level <= 2 else "NestJS"
            db_tech = "Prisma + PostgreSQL"
        elif language == "rust":
            api_framework = "Axum" if level >= 3 else "Actix-web"
            db_tech = "SQLx + PostgreSQL"
        elif language == "go":
            api_framework = "Gin" if level <= 3 else "Echo"
            db_tech = "GORM + PostgreSQL"
        else:  # nushell
            api_framework = "Custom HTTP server"
            db_tech = "JSON/CSV files"
        
        complexity_features = {
            1: ["Basic CRUD operations", "JSON responses"],
            2: ["Input validation", "Error handling", "Basic authentication"],
            3: ["Database integration", "Middleware", "JWT authentication"],
            4: ["Rate limiting", "Caching", "Background tasks"],
            5: ["Microservices", "Event sourcing", "Performance monitoring"]
        }
        
        return f"""# {language.title()} API Development Evaluation - Level {level} ({complexity['name']})

## Task Description
Develop a RESTful API in {language.title()} using {api_framework} that demonstrates {complexity['description'].lower()}.

**API Specification:**
Build a {'simple' if level <= 2 else 'comprehensive'} API for {'basic resource management' if level <= 2 else 'a complete application domain'}.

**Required Features:**
{chr(10).join(f"- {feature}" for feature in complexity_features[level])}

**Technical Stack:**
- Language: {language.title()}
- Framework: {api_framework}
- Database: {db_tech}
- Testing: {config['testing'][0] if config['testing'] else 'Manual testing'}
- Package Manager: {config['package_manager']}

**API Endpoints:**
{'- GET, POST /resources' if level <= 2 else '- Full CRUD with complex queries'}
{'- Authentication endpoints' if level >= 3 else ''}
{'- Admin and monitoring endpoints' if level >= 4 else ''}

## Language-Specific Requirements
Implement using {language.title()} best practices:
- {config['patterns'][0]}
- {config['patterns'][1] if len(config['patterns']) > 1 else 'Error handling'}
{'- ' + config['patterns'][2] if len(config['patterns']) > 2 and level >= 3 else ''}

## Evaluation Criteria
1. **API Design** (30%): RESTful principles and endpoint design
2. **Code Quality** (25%): {language.title()} conventions and structure
3. **Functionality** (20%): Feature completeness and reliability
4. **Security** (15%): Authentication, validation, and best practices
5. **Performance** (10%): Response times and scalability

## Expected Deliverables
- API implementation with all endpoints
- Database schema and migrations
- Comprehensive test suite
- API documentation (OpenAPI/Swagger)
{'- Performance benchmarks' if level >= 4 else ''}
{'- Security audit report' if level >= 5 else ''}

## Complexity Level: {level}/5 ({complexity['name']})
**Focus**: {complexity['description']}
"""

    def create_cli_prompt(self, language: str, level: int, config: Dict, complexity: Dict) -> str:
        """Create CLI tool prompt specific to language"""
        
        cli_features = {
            1: ["Basic command execution", "Simple output"],
            2: ["Command-line arguments", "Help text", "Error messages"],
            3: ["Subcommands", "Configuration files", "Interactive mode"],
            4: ["Plugin system", "Progress indicators", "Advanced output formatting"],
            5: ["Shell completion", "Update mechanism", "Comprehensive documentation"]
        }
        
        return f"""# {language.title()} CLI Tool Evaluation - Level {level} ({complexity['name']})

## Task Description
Create a command-line tool in {language.title()} that demonstrates {complexity['description'].lower()}.

**Tool Purpose:**
Build a CLI tool for {'basic file operations' if level <= 2 else 'complex workflow automation'}.

**Required Features:**
{chr(10).join(f"- {feature}" for feature in cli_features[level])}

**Technical Requirements:**
- Language: {language.title()}
- Package Manager: {config['package_manager']}
- Testing: {config['testing'][0] if config['testing'] else 'Manual testing'}
- Cross-platform compatibility

## Language-Specific Implementation
Use {language.title()} strengths:
- {config['patterns'][0]}
- {config['patterns'][1] if len(config['patterns']) > 1 else 'Standard patterns'}
{'- ' + config['patterns'][2] if len(config['patterns']) > 2 and level >= 3 else ''}

## Command Structure
{'Simple single command' if level <= 2 else 'Complex command hierarchy with subcommands'}

## Evaluation Criteria
1. **User Experience** (30%): Intuitive interface and helpful output
2. **Code Quality** (25%): Clean, maintainable {language.title()} code
3. **Functionality** (20%): Feature completeness and reliability
4. **Documentation** (15%): Help text, usage examples, man pages
5. **Cross-platform** (10%): Works on different operating systems

## Expected Deliverables
- CLI tool implementation
- Comprehensive test suite
- User documentation and examples
- Installation instructions
{'- Shell completion scripts' if level >= 4 else ''}
{'- Performance benchmarks' if level >= 5 else ''}

## Complexity Level: {level}/5 ({complexity['name']})
**Focus**: {complexity['description']}
"""

    def create_webapp_prompt(self, language: str, level: int, config: Dict, complexity: Dict) -> str:
        """Create web application prompt specific to language"""
        
        if language == "python":
            webapp_stack = "FastAPI + Jinja2 templates"
        elif language == "typescript":
            webapp_stack = "React/Vue + TypeScript"
        elif language == "rust":
            webapp_stack = "Axum + WASM frontend"
        elif language == "go":
            webapp_stack = "Echo + HTML templates"
        else:  # nushell
            webapp_stack = "Custom web server + static files"
        
        return f"""# {language.title()} Web Application Evaluation - Level {level} ({complexity['name']})

## Task Description
Develop a full-stack web application in {language.title()} using {webapp_stack}.

**Application Type:**
{'Simple static site' if level <= 2 else 'Dynamic web application with database'}

**Technical Stack:**
- Backend: {language.title()} with {config['frameworks'][0] if config['frameworks'] else 'standard library'}
- Frontend: {webapp_stack.split(' + ')[1] if ' + ' in webapp_stack else 'HTML/CSS/JS'}
- Database: {'File-based' if level <= 2 else 'PostgreSQL/SQLite'}

## Language-Specific Features
Leverage {language.title()} capabilities:
- {config['patterns'][0]}
- {config['patterns'][1] if len(config['patterns']) > 1 else 'Web patterns'}

## Evaluation Criteria
1. **User Interface** (25%): Design and user experience
2. **Backend Quality** (25%): {language.title()} best practices
3. **Functionality** (25%): Feature completeness
4. **Security** (15%): Authentication and data protection
5. **Performance** (10%): Load times and responsiveness

## Complexity Level: {level}/5 ({complexity['name']})
**Focus**: {complexity['description']}
"""

    def create_data_processing_prompt(self, language: str, level: int, config: Dict, complexity: Dict) -> str:
        """Create data processing prompt specific to language"""
        
        return f"""# {language.title()} Data Processing Evaluation - Level {level} ({complexity['name']})

## Task Description
Implement a data processing pipeline in {language.title()} that demonstrates {complexity['description'].lower()}.

**Data Processing Requirements:**
{'Basic file reading and simple transformations' if level <= 2 else 'Complex ETL pipeline with multiple data sources'}

**Language-Specific Approach:**
Use {language.title()} strengths for data processing:
- {config['patterns'][0]}
- {config['patterns'][1] if len(config['patterns']) > 1 else 'Data structures'}

## Evaluation Criteria
1. **Data Handling** (30%): Correct processing and transformations
2. **Code Quality** (25%): {language.title()} best practices
3. **Performance** (20%): Efficiency and memory usage
4. **Error Handling** (15%): Robust error management
5. **Documentation** (10%): Clear usage instructions

## Complexity Level: {level}/5 ({complexity['name']})
**Focus**: {complexity['description']}
"""

    def create_refactoring_prompt(self, language: str, level: int, config: Dict, complexity: Dict) -> str:
        """Create refactoring prompt specific to language"""
        
        return f"""# {language.title()} Code Refactoring Evaluation - Level {level} ({complexity['name']})

## Task Description
Refactor provided {language.title()} code to improve quality while maintaining functionality.

**Refactoring Goals:**
{'Basic cleanup and formatting' if level <= 2 else 'Architectural improvements and pattern application'}

**Language-Specific Improvements:**
Apply {language.title()} best practices:
- {config['patterns'][0]}
- {config['patterns'][1] if len(config['patterns']) > 1 else 'Code organization'}

## Evaluation Criteria
1. **Code Quality** (35%): Improved structure and readability
2. **Functionality** (25%): Preserved behavior and features
3. **Performance** (20%): Optimization improvements
4. **Maintainability** (20%): Long-term code health

## Complexity Level: {level}/5 ({complexity['name']})
**Focus**: {complexity['description']}
"""

    def create_generic_prompt(self, language: str, category: str, level: int, config: Dict, complexity: Dict) -> str:
        """Create a generic prompt template"""
        
        return f"""# {language.title()} {category.replace('-', ' ').title()} Evaluation - Level {level} ({complexity['name']})

## Task Description
Implement a {category.replace('-', ' ')} solution in {language.title()} that demonstrates {complexity['description'].lower()}.

**Technical Requirements:**
- Language: {language.title()}
- Package Manager: {config['package_manager']}
- Testing: {config['testing'][0] if config['testing'] else 'Manual testing'}

## Language-Specific Patterns
Apply these {language.title()} patterns:
- {config['patterns'][0]}
- {config['patterns'][1] if len(config['patterns']) > 1 else 'Standard conventions'}

## Evaluation Criteria
1. **Implementation Quality** (30%): {language.title()}-specific best practices
2. **Functionality** (25%): Feature completeness and correctness
3. **Code Organization** (20%): Structure and maintainability
4. **Documentation** (15%): Clear usage instructions
5. **Testing** (10%): Test coverage and quality

## Expected Deliverables
- Complete implementation
- Test suite
- Documentation
- Usage examples

## Complexity Level: {level}/5 ({complexity['name']})
**Focus**: {complexity['description']}
"""

    def generate_prompt_index(self):
        """Generate an index file for all prompts"""
        index_content = "# Language-Specific Evaluation Prompts Index\n\n"
        index_content += "This directory contains tailored evaluation prompts for each programming language.\n\n"
        
        for language in self.language_configs.keys():
            index_content += f"## {language.title()}\n\n"
            config = self.language_configs[language]
            index_content += f"**Frameworks**: {', '.join(config['frameworks'])}\n"
            index_content += f"**Testing**: {', '.join(config['testing'])}\n"
            index_content += f"**Tools**: {', '.join(config['tools'])}\n"
            index_content += f"**Key Patterns**: {', '.join(config['patterns'])}\n\n"
            
            for category in self.categories.keys():
                index_content += f"- **{self.categories[category]}**: `{language}/{category}/`\n"
            index_content += "\n"
        
        with open(self.output_dir / "README.md", 'w') as f:
            f.write(index_content)
        
        print("📚 Generated prompt index: README.md")

def main():
    parser = argparse.ArgumentParser(description='Generate Language-Specific Evaluation Prompts')
    parser.add_argument('--language', help='Generate prompts for specific language only')
    parser.add_argument('--output-dir', default='/workspace/agentic-eval/prompts',
                       help='Output directory for prompts')
    
    args = parser.parse_args()
    
    generator = LanguagePromptGenerator(args.output_dir)
    
    if args.language:
        if args.language in generator.language_configs:
            generator.generate_language_prompts(args.language)
        else:
            print(f"❌ Unknown language: {args.language}")
            print(f"Available languages: {', '.join(generator.language_configs.keys())}")
    else:
        generator.generate_all_prompts()
        generator.generate_prompt_index()

if __name__ == "__main__":
    main()