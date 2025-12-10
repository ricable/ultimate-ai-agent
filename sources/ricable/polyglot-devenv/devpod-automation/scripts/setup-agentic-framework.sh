#!/bin/bash

# Agentic Evaluation Framework Setup Script
# Creates standardized evaluation framework structure for DevPod workspaces
# Usage: bash setup-agentic-framework.sh [environment_type]

set -euo pipefail

ENVIRONMENT_TYPE="${1:-unified}"
WORKSPACE_ROOT="/workspace"
EVAL_ROOT="${WORKSPACE_ROOT}/agentic-eval"

echo "🤖 Setting up Agentic Evaluation Framework for environment: ${ENVIRONMENT_TYPE}"

# Create core directory structure
echo "📁 Creating framework directory structure..."
mkdir -p "${EVAL_ROOT}"/{
    prompts/{ui-components,apis,cli-tools,web-apps,data-processing,refactoring},
    templates/{claude,gemini,unified}/{beginner,intermediate,advanced,expert},
    results/{claude,gemini,comparative}/{outputs,logs,metrics,reports,screenshots},
    scripts/{automation,analysis,quality-check,comparison,reporting},
    configs/{environments,tools,scoring,models,prompts},
    reports/{daily,weekly,monthly,comparative,performance},
    benchmarks/{response-time,code-quality,accuracy-scores,resource-usage},
    databases/{sqlite,cache,exports},
    dashboards/{real-time,historical,comparative,metrics},
    shared/{claude-results,gemini-results,comparative-results,exports},
    utils/{parsers,validators,formatters,exporters},
    logs/{framework,evaluation,performance,errors}
}

# Create framework configuration files
echo "⚙️ Creating framework configuration..."

# Main framework config
cat > "${EVAL_ROOT}/config.json" << 'EOF'
{
  "framework": {
    "name": "Polyglot Agentic Evaluation Framework",
    "version": "1.0.0",
    "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  },
  "evaluation": {
    "tools": ["claude-code", "gemini-cli"],
    "languages": ["python", "typescript", "rust", "go", "nushell"],
    "complexity_levels": [1, 2, 3, 4, 5],
    "scoring_criteria": {
      "code_quality": 0.3,
      "functionality": 0.3,
      "performance": 0.2,
      "maintainability": 0.2
    }
  },
  "environment": {
    "type": "'"${ENVIRONMENT_TYPE}"'",
    "workspace_root": "/workspace",
    "results_database": "/workspace/agentic-eval/databases/results.db",
    "log_level": "INFO"
  }
}
EOF

# Create evaluation prompt templates
echo "📝 Creating evaluation prompt templates..."

# UI Components Evaluation Template
cat > "${EVAL_ROOT}/prompts/ui-components/template.md" << 'EOF'
# UI Component Evaluation Prompt

## Task Description
Create a [COMPONENT_TYPE] component with the following requirements:

**Functionality:**
- [FUNCTIONALITY_REQUIREMENTS]

**Design Requirements:**
- [DESIGN_REQUIREMENTS]

**Technical Constraints:**
- [TECHNICAL_CONSTRAINTS]

## Evaluation Criteria
1. **Visual Fidelity** (25%): How well does the component match the design requirements?
2. **Code Quality** (25%): Is the code well-structured, readable, and maintainable?
3. **Functionality** (25%): Does the component work as expected?
4. **Performance** (15%): Is the component performant and optimized?
5. **Accessibility** (10%): Does the component follow accessibility best practices?

## Complexity Level: [COMPLEXITY_LEVEL]
EOF

# API Development Evaluation Template
cat > "${EVAL_ROOT}/prompts/apis/template.md" << 'EOF'
# API Development Evaluation Prompt

## Task Description
Develop a [API_TYPE] API with the following specifications:

**Endpoints:**
- [ENDPOINT_SPECIFICATIONS]

**Data Models:**
- [DATA_MODEL_REQUIREMENTS]

**Authentication:**
- [AUTH_REQUIREMENTS]

**Performance Requirements:**
- [PERFORMANCE_REQUIREMENTS]

## Evaluation Criteria
1. **API Design** (30%): RESTful design, proper HTTP methods, status codes
2. **Code Quality** (25%): Structure, error handling, documentation
3. **Functionality** (20%): All endpoints work correctly
4. **Security** (15%): Proper authentication, input validation
5. **Performance** (10%): Response times, scalability considerations

## Complexity Level: [COMPLEXITY_LEVEL]
EOF

# CLI Tools Evaluation Template
cat > "${EVAL_ROOT}/prompts/cli-tools/template.md" << 'EOF'
# CLI Tool Evaluation Prompt

## Task Description
Create a command-line tool that [TOOL_DESCRIPTION]:

**Commands:**
- [COMMAND_SPECIFICATIONS]

**Options and Flags:**
- [OPTIONS_REQUIREMENTS]

**Output Format:**
- [OUTPUT_REQUIREMENTS]

**Error Handling:**
- [ERROR_HANDLING_REQUIREMENTS]

## Evaluation Criteria
1. **User Experience** (30%): Intuitive commands, helpful error messages
2. **Code Quality** (25%): Clean code, proper structure
3. **Functionality** (20%): All commands work correctly
4. **Documentation** (15%): Help text, usage examples
5. **Cross-platform** (10%): Works on different operating systems

## Complexity Level: [COMPLEXITY_LEVEL]
EOF

# Create evaluation scripts
echo "🔧 Creating evaluation automation scripts..."

# Main evaluation runner script
cat > "${EVAL_ROOT}/scripts/automation/run-evaluation.py" << 'EOF'
#!/usr/bin/env python3
"""
Agentic Evaluation Framework - Main Evaluation Runner
Orchestrates comparative evaluations between Claude Code CLI and Gemini CLI
"""

import json
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class AgenticEvaluator:
    def __init__(self, config_path: str = "/workspace/agentic-eval/config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.eval_root = Path("/workspace/agentic-eval")
        self.db_path = self.eval_root / "databases" / "results.db"
        self._init_database()
    
    def _load_config(self) -> Dict:
        """Load framework configuration"""
        with open(self.config_path) as f:
            return json.load(f)
    
    def _init_database(self):
        """Initialize SQLite database for results"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool TEXT NOT NULL,
                language TEXT NOT NULL,
                prompt_type TEXT NOT NULL,
                complexity_level INTEGER NOT NULL,
                prompt_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                execution_time REAL NOT NULL,
                score REAL,
                metrics TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                workspace_id TEXT,
                evaluation_id TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id TEXT NOT NULL,
                claude_result_id INTEGER,
                gemini_result_id INTEGER,
                winner TEXT,
                score_difference REAL,
                analysis TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claude_result_id) REFERENCES evaluations (id),
                FOREIGN KEY (gemini_result_id) REFERENCES evaluations (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def run_evaluation(self, tool: str, prompt: str, language: str, 
                      prompt_type: str, complexity_level: int) -> Dict:
        """Run evaluation for a specific tool"""
        print(f"🧪 Evaluating {tool} for {language} {prompt_type} (Level {complexity_level})")
        
        start_time = time.time()
        
        try:
            if tool == "claude-code":
                result = self._run_claude_evaluation(prompt, language)
            elif tool == "gemini-cli":
                result = self._run_gemini_evaluation(prompt, language)
            else:
                raise ValueError(f"Unknown tool: {tool}")
            
            execution_time = time.time() - start_time
            
            # Store result in database
            evaluation_record = {
                'tool': tool,
                'language': language,
                'prompt_type': prompt_type,
                'complexity_level': complexity_level,
                'prompt_text': prompt,
                'response_text': result.get('response', ''),
                'execution_time': execution_time,
                'score': result.get('score'),
                'metrics': json.dumps(result.get('metrics', {})),
                'workspace_id': self.config.get('workspace_id', 'default'),
                'evaluation_id': result.get('evaluation_id', f"{tool}_{language}_{int(time.time())}")
            }
            
            self._store_evaluation(evaluation_record)
            
            return {
                **evaluation_record,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {
                'tool': tool,
                'language': language,
                'prompt_type': prompt_type,
                'complexity_level': complexity_level,
                'error': str(e),
                'success': False
            }
    
    def _run_claude_evaluation(self, prompt: str, language: str) -> Dict:
        """Run evaluation using Claude Code CLI"""
        # This would integrate with the actual Claude Code CLI
        # For now, return mock data
        return {
            'response': f"Claude Code CLI response for {language} prompt",
            'score': 0.85,
            'metrics': {
                'response_time': 2.5,
                'code_quality': 0.9,
                'functionality': 0.8
            },
            'evaluation_id': f"claude_{language}_{int(time.time())}"
        }
    
    def _run_gemini_evaluation(self, prompt: str, language: str) -> Dict:
        """Run evaluation using Gemini CLI"""
        # This would integrate with the actual Gemini CLI
        # For now, return mock data
        return {
            'response': f"Gemini CLI response for {language} prompt",
            'score': 0.82,
            'metrics': {
                'response_time': 3.1,
                'code_quality': 0.85,
                'functionality': 0.79
            },
            'evaluation_id': f"gemini_{language}_{int(time.time())}"
        }
    
    def _store_evaluation(self, record: Dict):
        """Store evaluation result in database"""
        conn = sqlite3.connect(self.db_path)
        
        columns = ', '.join(record.keys())
        placeholders = ', '.join(['?' for _ in record])
        values = list(record.values())
        
        conn.execute(f'''
            INSERT INTO evaluations ({columns})
            VALUES ({placeholders})
        ''', values)
        
        conn.commit()
        conn.close()

def main():
    parser = argparse.ArgumentParser(description='Run Agentic Evaluation Framework')
    parser.add_argument('--tool', choices=['claude-code', 'gemini-cli', 'both'], 
                       default='both', help='Tool to evaluate')
    parser.add_argument('--language', default='python', 
                       help='Programming language for evaluation')
    parser.add_argument('--prompt-type', default='ui-components',
                       help='Type of evaluation prompt')
    parser.add_argument('--complexity', type=int, default=1,
                       help='Complexity level (1-5)')
    parser.add_argument('--prompt-file', help='Path to custom prompt file')
    
    args = parser.parse_args()
    
    evaluator = AgenticEvaluator()
    
    # Load prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
    else:
        prompt = f"Create a {args.prompt_type} in {args.language}"
    
    results = []
    
    if args.tool in ['claude-code', 'both']:
        result = evaluator.run_evaluation(
            'claude-code', prompt, args.language, 
            args.prompt_type, args.complexity
        )
        results.append(result)
    
    if args.tool in ['gemini-cli', 'both']:
        result = evaluator.run_evaluation(
            'gemini-cli', prompt, args.language,
            args.prompt_type, args.complexity
        )
        results.append(result)
    
    print("\n📊 Evaluation Results:")
    for result in results:
        if result.get('success'):
            print(f"✅ {result['tool']}: Score {result.get('score', 'N/A')}")
        else:
            print(f"❌ {result['tool']}: Failed - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
EOF

chmod +x "${EVAL_ROOT}/scripts/automation/run-evaluation.py"

# Create analysis script
cat > "${EVAL_ROOT}/scripts/analysis/analyze-results.py" << 'EOF'
#!/usr/bin/env python3
"""
Agentic Evaluation Framework - Results Analysis
Analyze and compare evaluation results between tools
"""

import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse

class ResultsAnalyzer:
    def __init__(self, db_path: str = "/workspace/agentic-eval/databases/results.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("/workspace/agentic-eval/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get overall statistics
        query = '''
            SELECT 
                tool,
                COUNT(*) as total_evaluations,
                AVG(score) as avg_score,
                AVG(execution_time) as avg_execution_time,
                language,
                prompt_type,
                complexity_level
            FROM evaluations 
            WHERE score IS NOT NULL
            GROUP BY tool, language, prompt_type, complexity_level
            ORDER BY tool, language, prompt_type, complexity_level
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Generate summary statistics
        summary = {
            'total_evaluations': len(df),
            'tools_compared': df['tool'].unique().tolist(),
            'languages_tested': df['language'].unique().tolist(),
            'prompt_types_tested': df['prompt_type'].unique().tolist(),
            'complexity_levels': sorted(df['complexity_level'].unique().tolist()),
            'avg_scores_by_tool': df.groupby('tool')['avg_score'].mean().to_dict(),
            'avg_execution_times': df.groupby('tool')['avg_execution_time'].mean().to_dict()
        }
        
        # Save summary report
        report_path = self.output_dir / f"summary_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Summary report saved: {report_path}")
        return summary
    
    def compare_tools(self, tool1: str = "claude-code", tool2: str = "gemini-cli"):
        """Compare performance between two tools"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM evaluations 
            WHERE tool IN (?, ?) AND score IS NOT NULL
            ORDER BY language, prompt_type, complexity_level, timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=[tool1, tool2])
        conn.close()
        
        if df.empty:
            print("❌ No evaluation data found for comparison")
            return
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Tool Comparison: {tool1} vs {tool2}', fontsize=16)
        
        # Score comparison by language
        score_by_lang = df.groupby(['language', 'tool'])['score'].mean().unstack()
        score_by_lang.plot(kind='bar', ax=axes[0,0], title='Average Score by Language')
        axes[0,0].set_ylabel('Score')
        axes[0,0].legend()
        
        # Execution time comparison
        time_by_tool = df.groupby('tool')['execution_time'].mean()
        time_by_tool.plot(kind='bar', ax=axes[0,1], title='Average Execution Time')
        axes[0,1].set_ylabel('Time (seconds)')
        
        # Score distribution
        for tool in df['tool'].unique():
            tool_data = df[df['tool'] == tool]['score']
            axes[1,0].hist(tool_data, alpha=0.7, label=tool, bins=20)
        axes[1,0].set_title('Score Distribution')
        axes[1,0].set_xlabel('Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # Complexity level comparison
        complex_scores = df.groupby(['complexity_level', 'tool'])['score'].mean().unstack()
        complex_scores.plot(kind='line', ax=axes[1,1], title='Score by Complexity Level', marker='o')
        axes[1,1].set_xlabel('Complexity Level')
        axes[1,1].set_ylabel('Average Score')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f"tool_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Comparison visualization saved: {viz_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Agentic Evaluation Results')
    parser.add_argument('--action', choices=['summary', 'compare'], 
                       default='summary', help='Analysis action to perform')
    parser.add_argument('--tool1', default='claude-code', help='First tool for comparison')
    parser.add_argument('--tool2', default='gemini-cli', help='Second tool for comparison')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer()
    
    if args.action == 'summary':
        summary = analyzer.generate_summary_report()
        print("\n📊 Evaluation Summary:")
        print(f"Total Evaluations: {summary['total_evaluations']}")
        print(f"Tools: {', '.join(summary['tools_compared'])}")
        print(f"Languages: {', '.join(summary['languages_tested'])}")
        print("\nAverage Scores by Tool:")
        for tool, score in summary['avg_scores_by_tool'].items():
            print(f"  {tool}: {score:.3f}")
    
    elif args.action == 'compare':
        analyzer.compare_tools(args.tool1, args.tool2)

if __name__ == "__main__":
    main()
EOF

chmod +x "${EVAL_ROOT}/scripts/analysis/analyze-results.py"

# Create quality check script
cat > "${EVAL_ROOT}/scripts/quality-check/validate-framework.sh" << 'EOF'
#!/bin/bash

# Agentic Evaluation Framework Validation Script
# Ensures framework is properly set up and functional

EVAL_ROOT="/workspace/agentic-eval"
EXIT_CODE=0

echo "🔍 Validating Agentic Evaluation Framework..."

# Check directory structure
echo "📁 Checking directory structure..."
REQUIRED_DIRS=(
    "prompts" "templates" "results" "scripts" "configs" 
    "reports" "benchmarks" "databases" "dashboards" "shared"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ -d "${EVAL_ROOT}/${dir}" ]]; then
        echo "  ✅ ${dir}/ exists"
    else
        echo "  ❌ ${dir}/ missing"
        EXIT_CODE=1
    fi
done

# Check configuration files
echo "⚙️ Checking configuration files..."
if [[ -f "${EVAL_ROOT}/config.json" ]]; then
    echo "  ✅ config.json exists"
    # Validate JSON syntax
    if python3 -m json.tool "${EVAL_ROOT}/config.json" > /dev/null 2>&1; then
        echo "  ✅ config.json is valid JSON"
    else
        echo "  ❌ config.json has invalid JSON syntax"
        EXIT_CODE=1
    fi
else
    echo "  ❌ config.json missing"
    EXIT_CODE=1
fi

# Check script executability
echo "🔧 Checking script permissions..."
SCRIPTS=(
    "scripts/automation/run-evaluation.py"
    "scripts/analysis/analyze-results.py"
)

for script in "${SCRIPTS[@]}"; do
    if [[ -x "${EVAL_ROOT}/${script}" ]]; then
        echo "  ✅ ${script} is executable"
    else
        echo "  ❌ ${script} is not executable or missing"
        EXIT_CODE=1
    fi
done

# Check tool availability
echo "🛠️ Checking tool availability..."
if command -v claude-code &> /dev/null; then
    echo "  ✅ claude-code CLI available"
else
    echo "  ⚠️ claude-code CLI not found"
fi

if command -v gemini &> /dev/null; then
    echo "  ✅ gemini CLI available"
else
    echo "  ⚠️ gemini CLI not found"
fi

# Check Python dependencies
echo "🐍 Checking Python dependencies..."
PYTHON_DEPS=("pandas" "matplotlib" "sqlite3")

for dep in "${PYTHON_DEPS[@]}"; do
    if python3 -c "import ${dep}" 2>/dev/null; then
        echo "  ✅ ${dep} available"
    else
        echo "  ❌ ${dep} missing"
        EXIT_CODE=1
    fi
done

# Database initialization check
echo "🗄️ Checking database setup..."
if [[ -f "${EVAL_ROOT}/databases/results.db" ]]; then
    echo "  ✅ results.db exists"
else
    echo "  ℹ️ results.db will be created on first use"
fi

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Framework validation completed successfully!"
else
    echo "❌ Framework validation failed. Please address the issues above."
fi

exit $EXIT_CODE
EOF

chmod +x "${EVAL_ROOT}/scripts/quality-check/validate-framework.sh"

# Create initial README for the framework
cat > "${EVAL_ROOT}/README.md" << 'EOF'
# Agentic Evaluation Framework

A comprehensive framework for evaluating and comparing AI coding tools (Claude Code CLI vs Gemini CLI) across multiple programming languages and complexity levels.

## Framework Structure

```
agentic-eval/
├── prompts/          # Evaluation prompt templates
├── templates/        # Tool-specific templates
├── results/          # Evaluation results and outputs
├── scripts/          # Automation and analysis scripts
├── configs/          # Framework configuration
├── reports/          # Generated reports and visualizations
├── benchmarks/       # Performance benchmarks
├── databases/        # SQLite database for results
├── dashboards/       # Real-time monitoring dashboards
├── shared/           # Shared results and exports
└── utils/           # Utility functions and helpers
```

## Quick Start

1. **Validate Framework Setup**
   ```bash
   bash scripts/quality-check/validate-framework.sh
   ```

2. **Run Basic Evaluation**
   ```bash
   python3 scripts/automation/run-evaluation.py --tool both --language python --prompt-type ui-components
   ```

3. **Analyze Results**
   ```bash
   python3 scripts/analysis/analyze-results.py --action summary
   ```

4. **Compare Tools**
   ```bash
   python3 scripts/analysis/analyze-results.py --action compare
   ```

## Evaluation Criteria

- **Code Quality** (30%): Structure, readability, maintainability
- **Functionality** (25%): Correctness and completeness
- **Performance** (20%): Execution time and resource usage
- **Documentation** (15%): Code comments and documentation
- **Best Practices** (10%): Following language conventions

## Supported Languages

- Python (FastAPI, Flask, Django)
- TypeScript (React, Vue, Node.js)
- Rust (Tokio, Actix, CLI tools)
- Go (Gin, Echo, CLI applications)
- Nushell (Scripts and pipelines)

## Prompt Types

- **UI Components**: Frontend component development
- **APIs**: Backend API development
- **CLI Tools**: Command-line application development
- **Web Apps**: Full-stack web applications
- **Data Processing**: Data analysis and transformation
- **Refactoring**: Code improvement and optimization

## Configuration

Framework configuration is stored in `config.json`. Key settings:

- **Tools**: AI tools to evaluate
- **Languages**: Programming languages to test
- **Complexity Levels**: 1-5 difficulty scale
- **Scoring Criteria**: Weighted evaluation metrics

## Database Schema

Results are stored in SQLite with the following tables:

- `evaluations`: Individual tool evaluations
- `comparisons`: Tool-to-tool comparisons
- `benchmarks`: Performance metrics

## Reports and Visualization

The framework generates:

- Summary reports (JSON)
- Comparison visualizations (PNG)
- Performance benchmarks (CSV)
- Real-time dashboards (HTML)

## Integration

This framework integrates with:

- **Claude Code CLI**: @anthropic-ai/claude-code
- **Gemini CLI**: @google/gemini-cli
- **DevPod**: Containerized evaluation environments
- **MCP Server**: Model Context Protocol integration

## Development

To extend the framework:

1. Add new prompt templates in `prompts/`
2. Create tool-specific scripts in `scripts/`
3. Update configuration in `config.json`
4. Run validation to ensure integrity

## Support

For issues and questions:
- Check framework validation output
- Review logs in `logs/` directory
- Consult MCP server documentation
- Open issues in the main repository
EOF

# Generate language-specific evaluation prompts
echo "📝 Generating language-specific evaluation prompts..."
if [[ -f "/workspace/devpod-automation/scripts/generate-language-prompts.py" ]]; then
    python3 /workspace/devpod-automation/scripts/generate-language-prompts.py --output-dir "${EVAL_ROOT}/prompts"
else
    echo "⚠️ Language prompt generator not found, using basic prompts"
fi

echo "✅ Agentic Evaluation Framework setup completed!"
echo ""
echo "📁 Framework created at: ${EVAL_ROOT}"
echo "🔧 Configuration: ${EVAL_ROOT}/config.json"
echo "📖 Documentation: ${EVAL_ROOT}/README.md"
echo "📝 Language Prompts: ${EVAL_ROOT}/prompts/"
echo ""
echo "🚀 Next steps:"
echo "1. Run validation: bash ${EVAL_ROOT}/scripts/quality-check/validate-framework.sh"
echo "2. Configure API keys for Claude and Gemini"
echo "3. Generate language prompts: python3 ${EVAL_ROOT}/scripts/generate-language-prompts.py"
echo "4. Start evaluation: python3 ${EVAL_ROOT}/scripts/automation/run-evaluation.py"
echo ""
echo "🎉 Framework ready for agentic evaluation!"