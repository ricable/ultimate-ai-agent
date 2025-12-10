#!/usr/bin/env python3
"""
Environment-Specific Prompt Adaptation Script
Adapts evaluation prompts based on the specific DevPod environment type
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import argparse

class EnvironmentPromptAdapter:
    def __init__(self, environment_type: str, eval_root: str = "/workspace/agentic-eval"):
        self.environment_type = environment_type
        self.eval_root = Path(eval_root)
        self.prompts_dir = self.eval_root / "prompts"
        
        # Environment-specific configurations
        self.environment_configs = {
            "unified": {
                "description": "Comparative evaluation across all languages and tools",
                "languages": ["python", "typescript", "rust", "go", "nushell"],
                "tools": ["claude-code", "gemini-cli"],
                "focus": "Cross-language comparison and tool performance analysis"
            },
            "claude": {
                "description": "Claude Code CLI focused evaluation",
                "languages": ["python", "typescript"],  # Primary languages for Claude
                "tools": ["claude-code"],
                "focus": "Claude-specific capabilities and optimization"
            },
            "gemini": {
                "description": "Gemini CLI focused evaluation", 
                "languages": ["python", "typescript"],  # Primary languages for Gemini
                "tools": ["gemini-cli"],
                "focus": "Gemini-specific capabilities and optimization"
            },
            "results": {
                "description": "Results analysis and visualization",
                "languages": ["python"],  # Python for data analysis
                "tools": ["analysis-tools"],
                "focus": "Data analysis, visualization, and reporting"
            }
        }

    def adapt_all_prompts(self):
        """Adapt all prompts for the current environment"""
        print(f"🔧 Adapting prompts for {self.environment_type} environment...")
        
        config = self.environment_configs.get(self.environment_type, self.environment_configs["unified"])
        
        # Create environment-specific prompt structure
        self.create_environment_structure(config)
        
        # Generate tool-specific prompts
        self.generate_tool_specific_prompts(config)
        
        # Create evaluation workflows
        self.create_evaluation_workflows(config)
        
        print(f"✅ Prompts adapted for {self.environment_type} environment!")

    def create_environment_structure(self, config: Dict):
        """Create environment-specific directory structure"""
        env_prompts_dir = self.prompts_dir / "environment-specific" / self.environment_type
        env_prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create language-specific directories for this environment
        for language in config["languages"]:
            lang_dir = env_prompts_dir / language
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy and adapt base prompts for this language
            base_lang_dir = self.prompts_dir / language
            if base_lang_dir.exists():
                for category_dir in base_lang_dir.iterdir():
                    if category_dir.is_dir():
                        env_category_dir = lang_dir / category_dir.name
                        env_category_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Copy and adapt prompts
                        for prompt_file in category_dir.glob("*.md"):
                            adapted_content = self.adapt_prompt_content(
                                prompt_file.read_text(), config, language, category_dir.name
                            )
                            env_prompt_file = env_category_dir / prompt_file.name
                            env_prompt_file.write_text(adapted_content)
                            
        print(f"  📁 Created environment structure for {len(config['languages'])} languages")

    def adapt_prompt_content(self, content: str, config: Dict, language: str, category: str) -> str:
        """Adapt prompt content for the specific environment"""
        
        # Add environment-specific header
        env_header = f"""# {self.environment_type.title()} Environment Evaluation
**Environment Focus**: {config['focus']}
**Tools**: {', '.join(config['tools'])}
**Evaluation Mode**: {self.environment_type}

---

"""
        
        # Add tool-specific instructions
        if self.environment_type == "claude":
            tool_instructions = """
## Claude Code CLI Specific Instructions
- Leverage Claude's advanced reasoning capabilities
- Focus on code quality and architectural decisions
- Utilize Claude's context awareness for complex problems
- Test edge cases and error handling thoroughly

"""
        elif self.environment_type == "gemini":
            tool_instructions = """
## Gemini CLI Specific Instructions  
- Leverage Gemini's multimodal capabilities where applicable
- Focus on performance optimization and efficiency
- Utilize Gemini's broad knowledge base
- Test scalability and resource usage

"""
        elif self.environment_type == "unified":
            tool_instructions = """
## Comparative Evaluation Instructions
- Run the same prompt with both Claude Code CLI and Gemini CLI
- Document differences in approach and output quality
- Measure response times and resource usage
- Compare code quality, functionality, and maintainability
- Provide detailed comparative analysis

"""
        else:  # results
            tool_instructions = """
## Results Analysis Instructions
- Focus on data processing and visualization
- Create comprehensive reports and dashboards
- Implement statistical analysis and metrics
- Generate comparative visualizations

"""
        
        # Add environment-specific evaluation criteria
        if self.environment_type == "unified":
            eval_criteria = """
## Comparative Evaluation Criteria
1. **Tool Performance** (20%): Response time and resource usage
2. **Code Quality** (25%): Structure, readability, best practices
3. **Functionality** (25%): Feature completeness and correctness
4. **Innovation** (15%): Creative solutions and approaches
5. **Documentation** (15%): Clarity and completeness

## Comparison Metrics
- **Response Time**: Time to generate complete solution
- **Code Lines**: Total lines of code generated
- **Test Coverage**: Percentage of code covered by tests
- **Error Handling**: Robustness of error management
- **Performance**: Execution speed and memory usage

"""
        else:
            eval_criteria = """
## Tool-Specific Evaluation Criteria
1. **Code Quality** (30%): Best practices and patterns
2. **Functionality** (25%): Feature completeness
3. **Performance** (20%): Optimization and efficiency
4. **Maintainability** (15%): Long-term code health
5. **Documentation** (10%): Usage and implementation docs

"""
        
        # Combine all parts
        adapted_content = env_header + tool_instructions + content + "\n" + eval_criteria
        
        return adapted_content

    def generate_tool_specific_prompts(self, config: Dict):
        """Generate prompts specific to the tools in this environment"""
        tools_dir = self.prompts_dir / "tool-specific" / self.environment_type
        tools_dir.mkdir(parents=True, exist_ok=True)
        
        for tool in config["tools"]:
            tool_dir = tools_dir / tool
            tool_dir.mkdir(parents=True, exist_ok=True)
            
            # Create tool-specific prompt templates
            self.create_tool_prompt_template(tool, tool_dir, config)
            
        print(f"  🔧 Generated tool-specific prompts for {len(config['tools'])} tools")

    def create_tool_prompt_template(self, tool: str, tool_dir: Path, config: Dict):
        """Create a prompt template for a specific tool"""
        
        if tool == "claude-code":
            template_content = """# Claude Code CLI Evaluation Template

## Tool Configuration
- Model: claude-3-5-sonnet-20241022
- Max Tokens: 4096
- Temperature: 0.1 (for consistent code generation)

## Claude-Specific Prompting Strategy
1. **Context Setting**: Provide comprehensive background and requirements
2. **Step-by-Step Reasoning**: Ask Claude to explain its approach
3. **Error Handling**: Request explicit error handling strategies
4. **Best Practices**: Emphasize language-specific best practices
5. **Testing**: Request comprehensive test coverage

## Prompt Template
```
You are an expert {language} developer. Please create a {task_type} that meets the following requirements:

[REQUIREMENTS]

Please follow these steps:
1. Analyze the requirements and explain your approach
2. Implement the solution using {language} best practices
3. Include comprehensive error handling
4. Add thorough test coverage
5. Provide usage documentation

Focus on code quality, maintainability, and performance.
```

## Evaluation Focus
- Code architecture and design patterns
- Error handling and edge cases
- Test coverage and quality
- Documentation completeness
- Performance considerations
"""
        
        elif tool == "gemini-cli":
            template_content = """# Gemini CLI Evaluation Template

## Tool Configuration
- Model: gemini-1.5-pro
- Max Tokens: 8192
- Temperature: 0.2 (balanced creativity and consistency)

## Gemini-Specific Prompting Strategy
1. **Clear Instructions**: Provide specific, actionable requirements
2. **Output Format**: Specify exact output format and structure
3. **Performance Focus**: Emphasize efficiency and optimization
4. **Practical Examples**: Include concrete use cases
5. **Validation**: Request self-validation of the solution

## Prompt Template
```
Create a {task_type} in {language} with the following specifications:

[REQUIREMENTS]

Requirements:
- Implement using {language} best practices
- Optimize for performance and memory usage
- Include input validation and error handling
- Provide clear documentation and examples
- Add unit tests for all major functionality

Please provide the complete implementation with explanations.
```

## Evaluation Focus
- Performance optimization
- Resource efficiency
- Input validation
- Practical usability
- Implementation completeness
"""
        
        else:  # analysis-tools or other
            template_content = """# Analysis Tools Evaluation Template

## Analysis Configuration
- Focus: Data processing and visualization
- Tools: Python, pandas, matplotlib, jupyter
- Output: Reports, dashboards, insights

## Analysis Prompting Strategy
1. **Data Understanding**: Analyze data structure and quality
2. **Methodology**: Choose appropriate analysis methods
3. **Visualization**: Create meaningful charts and graphs
4. **Insights**: Extract actionable insights
5. **Reporting**: Present findings clearly

## Analysis Template
```
Analyze the evaluation results data and create:

[REQUIREMENTS]

Please provide:
1. Data quality assessment
2. Statistical analysis and insights
3. Comparative visualizations
4. Performance metrics dashboard
5. Executive summary report

Focus on actionable insights and clear presentation.
```

## Evaluation Focus
- Data analysis accuracy
- Visualization effectiveness
- Insight quality
- Report clarity
- Dashboard usability
"""
        
        with open(tool_dir / "template.md", 'w') as f:
            f.write(template_content)

    def create_evaluation_workflows(self, config: Dict):
        """Create evaluation workflows for this environment"""
        workflows_dir = self.prompts_dir / "workflows" / self.environment_type
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Create workflow configuration
        workflow_config = {
            "environment": self.environment_type,
            "description": config["description"],
            "languages": config["languages"],
            "tools": config["tools"],
            "focus": config["focus"],
            "workflow_steps": self.get_workflow_steps(config)
        }
        
        with open(workflows_dir / "config.json", 'w') as f:
            json.dump(workflow_config, f, indent=2)
        
        # Create workflow scripts
        self.create_workflow_scripts(workflows_dir, config)
        
        print(f"  🔄 Created evaluation workflows for {self.environment_type}")

    def get_workflow_steps(self, config: Dict) -> List[str]:
        """Get workflow steps for the environment type"""
        
        if self.environment_type == "unified":
            return [
                "Initialize comparative evaluation environment",
                "Load evaluation prompts for all languages",
                "Execute prompts with Claude Code CLI",
                "Execute prompts with Gemini CLI", 
                "Collect and store results",
                "Perform comparative analysis",
                "Generate comparison reports",
                "Create visualization dashboards"
            ]
        elif self.environment_type in ["claude", "gemini"]:
            tool_name = "Claude Code CLI" if self.environment_type == "claude" else "Gemini CLI"
            return [
                f"Initialize {tool_name} evaluation environment",
                "Load tool-specific evaluation prompts",
                f"Execute prompts with {tool_name}",
                "Collect and analyze results",
                "Measure performance metrics",
                f"Generate {tool_name} performance report",
                "Create optimization recommendations"
            ]
        else:  # results
            return [
                "Load evaluation results from all environments",
                "Perform data quality assessment",
                "Execute statistical analysis",
                "Create comparative visualizations",
                "Generate executive dashboards", 
                "Produce final evaluation report"
            ]

    def create_workflow_scripts(self, workflows_dir: Path, config: Dict):
        """Create workflow automation scripts"""
        
        # Create main workflow script
        workflow_script = f"""#!/usr/bin/env python3
\"\"\"
{self.environment_type.title()} Environment Evaluation Workflow
Automated workflow for {config['description']}
\"\"\"

import sys
import json
import subprocess
from pathlib import Path

def main():
    print(f"🚀 Starting {self.environment_type} evaluation workflow...")
    
    # Load workflow configuration
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        workflow_config = json.load(f)
    
    print(f"📋 Workflow: {{workflow_config['description']}}")
    print(f"🔧 Tools: {{', '.join(workflow_config['tools'])}}")
    print(f"🌐 Languages: {{', '.join(workflow_config['languages'])}}")
    
    # Execute workflow steps
    for i, step in enumerate(workflow_config['workflow_steps'], 1):
        print(f"\\n📍 Step {{i}}/{{len(workflow_config['workflow_steps'])}}: {{step}}")
        
        # Here you would implement the actual step execution
        # For now, we'll just simulate the steps
        print(f"   ✅ Completed: {{step}}")
    
    print(f"\\n🎉 {{self.environment_type.title()}} evaluation workflow completed!")

if __name__ == "__main__":
    main()
"""
        
        with open(workflows_dir / "run_workflow.py", 'w') as f:
            f.write(workflow_script)
        
        # Make it executable
        os.chmod(workflows_dir / "run_workflow.py", 0o755)

    def create_environment_readme(self):
        """Create README for the environment-specific prompts"""
        config = self.environment_configs.get(self.environment_type, {})
        
        readme_content = f"""# {self.environment_type.title()} Environment Evaluation Prompts

## Environment Overview
**Type**: {self.environment_type}
**Description**: {config.get('description', 'Custom evaluation environment')}
**Focus**: {config.get('focus', 'General evaluation')}

## Supported Languages
{chr(10).join(f"- {lang.title()}" for lang in config.get('languages', []))}

## Available Tools
{chr(10).join(f"- {tool}" for tool in config.get('tools', []))}

## Directory Structure
```
{self.environment_type}/
├── environment-specific/     # Environment-adapted prompts
├── tool-specific/           # Tool-specific templates
├── workflows/              # Evaluation workflows
└── README.md              # This file
```

## Usage

### Running Evaluations
```bash
# Run environment-specific workflow
python3 workflows/{self.environment_type}/run_workflow.py

# Generate language-specific prompts
python3 scripts/generate-language-prompts.py --language python

# Run individual evaluation
python3 scripts/automation/run-evaluation.py --tool {config.get('tools', ['both'])[0]} --language python
```

### Prompt Structure
Each prompt includes:
- Environment-specific instructions
- Tool-specific guidance
- Language-appropriate patterns
- Evaluation criteria
- Expected deliverables

## Evaluation Focus
{config.get('focus', 'General evaluation and comparison')}

## Next Steps
1. Review environment-specific prompts
2. Configure API keys for tools
3. Run validation workflow
4. Execute evaluations
5. Analyze results
"""
        
        readme_path = self.prompts_dir / "environment-specific" / self.environment_type / "README.md"
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)

def main():
    parser = argparse.ArgumentParser(description='Adapt Evaluation Prompts for Environment')
    parser.add_argument('environment', choices=['unified', 'claude', 'gemini', 'results'],
                       help='Environment type to adapt prompts for')
    parser.add_argument('--eval-root', default='/workspace/agentic-eval',
                       help='Evaluation framework root directory')
    
    args = parser.parse_args()
    
    adapter = EnvironmentPromptAdapter(args.environment, args.eval_root)
    adapter.adapt_all_prompts()
    adapter.create_environment_readme()
    
    print(f"\\n✅ Environment adaptation completed for {args.environment}")
    print(f"📁 Adapted prompts available at: {adapter.prompts_dir}/environment-specific/{args.environment}/")

if __name__ == "__main__":
    main()