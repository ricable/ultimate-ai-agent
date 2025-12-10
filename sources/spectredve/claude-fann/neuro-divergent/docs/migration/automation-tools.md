# Migration Automation Tools: Automated Migration Utilities

This guide provides comprehensive automation tools and utilities to streamline the migration from Python NeuralForecast to Rust neuro-divergent.

## Table of Contents

1. [Code Analysis Tools](#code-analysis-tools)
2. [Automated Conversion Scripts](#automated-conversion-scripts)
3. [Validation and Testing Tools](#validation-and-testing-tools)
4. [Performance Benchmarking](#performance-benchmarking)
5. [Configuration Migration](#configuration-migration)
6. [Deployment Automation](#deployment-automation)
7. [Monitoring and Reporting](#monitoring-and-reporting)
8. [CI/CD Integration](#cicd-integration)

## Code Analysis Tools

### Python Codebase Analyzer

**Script: `analyze_python_code.py`**
```python
#!/usr/bin/env python3
"""
Analyze Python NeuralForecast codebase for migration planning.
"""

import ast
import os
import json
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, asdict

@dataclass
class MigrationAnalysis:
    models_used: Set[str]
    parameters_used: Dict[str, Set[str]]
    data_operations: List[str]
    dependencies: Set[str]
    complexity_score: int
    estimated_migration_time: str

class NeuralForecastAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.models_used = set()
        self.parameters_used = {}
        self.data_operations = []
        self.dependencies = set()
        self.function_calls = []
        
    def visit_Import(self, node):
        for alias in node.names:
            if 'neuralforecast' in alias.name:
                self.dependencies.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module and 'neuralforecast' in node.module:
            for alias in node.names:
                self.dependencies.add(f"{node.module}.{alias.name}")
                if 'models' in node.module:
                    self.models_used.add(alias.name)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Track model instantiations
        if isinstance(node.func, ast.Name):
            if node.func.id in ['LSTM', 'NBEATS', 'TFT', 'MLP', 'DLinear']:
                self.models_used.add(node.func.id)
                # Extract parameters
                params = set()
                for keyword in node.keywords:
                    params.add(keyword.arg)
                self.parameters_used[node.func.id] = params
        
        # Track data operations
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ['fit', 'predict', 'cross_validation', 'groupby', 'fillna']:
                self.data_operations.append(method_name)
        
        self.generic_visit(node)

def analyze_file(file_path: Path) -> MigrationAnalysis:
    """Analyze a single Python file for migration complexity."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return None
    
    analyzer = NeuralForecastAnalyzer()
    analyzer.visit(tree)
    
    # Calculate complexity score
    complexity = (
        len(analyzer.models_used) * 10 +
        len(analyzer.data_operations) * 2 +
        sum(len(params) for params in analyzer.parameters_used.values())
    )
    
    # Estimate migration time
    if complexity < 20:
        time_estimate = "1-2 days"
    elif complexity < 50:
        time_estimate = "3-5 days"
    elif complexity < 100:
        time_estimate = "1-2 weeks"
    else:
        time_estimate = "2+ weeks"
    
    return MigrationAnalysis(
        models_used=analyzer.models_used,
        parameters_used=analyzer.parameters_used,
        data_operations=analyzer.data_operations,
        dependencies=analyzer.dependencies,
        complexity_score=complexity,
        estimated_migration_time=time_estimate
    )

def analyze_project(project_path: str) -> Dict:
    """Analyze entire project for migration planning."""
    project_path = Path(project_path)
    results = {}
    total_complexity = 0
    all_models = set()
    all_dependencies = set()
    
    for py_file in project_path.rglob('*.py'):
        if py_file.name.startswith('.'):
            continue
            
        analysis = analyze_file(py_file)
        if analysis:
            results[str(py_file.relative_to(project_path))] = asdict(analysis)
            total_complexity += analysis.complexity_score
            all_models.update(analysis.models_used)
            all_dependencies.update(analysis.dependencies)
    
    # Generate migration plan
    migration_plan = {
        'project_complexity': total_complexity,
        'models_to_migrate': sorted(all_models),
        'dependencies_found': sorted(all_dependencies),
        'recommended_strategy': (
            'gradual' if total_complexity > 100 else 
            'big_bang' if total_complexity < 50 else 
            'side_by_side'
        ),
        'file_analyses': results,
        'migration_priorities': prioritize_files(results)
    }
    
    return migration_plan

def prioritize_files(analyses: Dict) -> List[Dict]:
    """Prioritize files for migration based on complexity and dependencies."""
    priorities = []
    
    for file_path, analysis in analyses.items():
        priority_score = analysis['complexity_score']
        
        # Boost priority for files with many models
        if len(analysis['models_used']) > 2:
            priority_score += 20
        
        # Boost priority for core functionality
        if any(op in analysis['data_operations'] for op in ['fit', 'predict']):
            priority_score += 15
        
        priorities.append({
            'file': file_path,
            'priority_score': priority_score,
            'complexity': analysis['complexity_score'],
            'models': list(analysis['models_used']),
            'recommended_order': 'high' if priority_score > 50 else 'medium' if priority_score > 20 else 'low'
        })
    
    return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_python_code.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    analysis = analyze_project(project_path)
    
    # Save analysis to file
    with open('migration_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"Analysis complete. Results saved to migration_analysis.json")
    print(f"Project complexity: {analysis['project_complexity']}")
    print(f"Models found: {', '.join(analysis['models_to_migrate'])}")
    print(f"Recommended strategy: {analysis['recommended_strategy']}")
```

### Dependency Mapper

**Script: `map_dependencies.py`**
```python
#!/usr/bin/env python3
"""
Map Python dependencies to Rust equivalents.
"""

import json
from typing import Dict, List

DEPENDENCY_MAPPING = {
    'pandas': 'polars',
    'numpy': 'ndarray (optional)',
    'scikit-learn': 'linfa (or custom implementation)',
    'matplotlib': 'plotters',
    'seaborn': 'plotters + custom styling',
    'plotly': 'plotly.js via WASM',
    'jupyter': 'evcxr_jupyter',
    'fastapi': 'axum or warp',
    'uvicorn': 'tokio runtime',
    'pydantic': 'serde',
    'requests': 'reqwest',
    'sqlalchemy': 'sqlx or diesel',
    'pytest': 'built-in testing + cargo test',
    'mlflow': 'mlflow-rust (community)',
    'wandb': 'wandb-rs (community)',
}

MODEL_MAPPING = {
    'LSTM': 'neuro_divergent::models::LSTM',
    'GRU': 'neuro_divergent::models::GRU',
    'MLP': 'neuro_divergent::models::MLP',
    'NBEATS': 'neuro_divergent::models::NBEATS',
    'TFT': 'neuro_divergent::models::TFT',
    'DLinear': 'neuro_divergent::models::DLinear',
    'NLinear': 'neuro_divergent::models::NLinear',
    'Autoformer': 'neuro_divergent::models::Autoformer',
    'Informer': 'neuro_divergent::models::Informer',
}

def generate_cargo_toml(dependencies: List[str], models: List[str]) -> str:
    """Generate Cargo.toml dependencies section."""
    cargo_deps = {
        'neuro-divergent': '"0.1.0"',
        'polars': '{ version = "0.33", features = ["lazy", "csv", "parquet"] }',
        'tokio': '{ version = "1.0", features = ["full"] }',
        'anyhow': '"1.0"',
        'serde': '{ version = "1.0", features = ["derive"] }',
    }
    
    # Add dependencies based on Python requirements
    for dep in dependencies:
        if 'fastapi' in dep or 'uvicorn' in dep:
            cargo_deps['axum'] = '"0.7"'
        elif 'requests' in dep:
            cargo_deps['reqwest'] = '{ version = "0.11", features = ["json"] }'
        elif 'sqlalchemy' in dep:
            cargo_deps['sqlx'] = '{ version = "0.7", features = ["runtime-tokio-rustls", "postgres"] }'
        elif 'pytest' in dep:
            cargo_deps['criterion'] = '"0.5"'
    
    toml_content = "[dependencies]\n"
    for name, version in sorted(cargo_deps.items()):
        toml_content += f"{name} = {version}\n"
    
    return toml_content

def create_migration_guide(analysis_file: str) -> str:
    """Create personalized migration guide based on analysis."""
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    guide = "# Personalized Migration Guide\n\n"
    
    # Project overview
    guide += f"## Project Overview\n"
    guide += f"- Complexity Score: {analysis['project_complexity']}\n"
    guide += f"- Recommended Strategy: {analysis['recommended_strategy']}\n"
    guide += f"- Models to Migrate: {', '.join(analysis['models_to_migrate'])}\n\n"
    
    # Dependencies mapping
    guide += "## Dependencies Migration\n\n"
    for dep in analysis['dependencies_found']:
        base_dep = dep.split('.')[0]
        if base_dep in DEPENDENCY_MAPPING:
            guide += f"- `{dep}` → `{DEPENDENCY_MAPPING[base_dep]}`\n"
    
    guide += "\n"
    
    # Model migration
    guide += "## Model Migration\n\n"
    for model in analysis['models_to_migrate']:
        if model in MODEL_MAPPING:
            guide += f"### {model}\n"
            guide += f"```rust\n"
            guide += f"use {MODEL_MAPPING[model]};\n\n"
            guide += f"let model = {model}::builder()\n"
            guide += f"    .horizon(12)\n"
            guide += f"    .input_size(24)\n"
            guide += f"    .build()?;\n"
            guide += f"```\n\n"
    
    # File migration priorities
    guide += "## Migration Priorities\n\n"
    for priority in analysis['migration_priorities'][:5]:  # Top 5 files
        guide += f"### {priority['recommended_order'].title()} Priority: `{priority['file']}`\n"
        guide += f"- Complexity: {priority['complexity']}\n"
        guide += f"- Models: {', '.join(priority['models'])}\n\n"
    
    return guide

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python map_dependencies.py <analysis_file>")
        sys.exit(1)
    
    analysis_file = sys.argv[1]
    guide = create_migration_guide(analysis_file)
    
    with open('personalized_migration_guide.md', 'w') as f:
        f.write(guide)
    
    print("Personalized migration guide created: personalized_migration_guide.md")
```

## Automated Conversion Scripts

### Configuration Converter

**Script: `convert_config.py`**
```python
#!/usr/bin/env python3
"""
Convert Python configuration to Rust TOML format.
"""

import yaml
import json
import toml
from pathlib import Path
from typing import Dict, Any

def convert_python_config_to_rust(python_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Python configuration dictionary to Rust format."""
    rust_config = {}
    
    # Data configuration
    if 'data' in python_config:
        data_config = python_config['data']
        rust_config['data'] = {
            'path': data_config.get('path', './data'),
            'frequency': convert_frequency(data_config.get('freq', 'D')),
        }
        
        if 'static_features' in data_config:
            rust_config['data']['static_features'] = data_config['static_features']
        if 'future_features' in data_config:
            rust_config['data']['future_features'] = data_config['future_features']
    
    # Models configuration
    if 'models' in python_config:
        rust_config['models'] = {}
        for model_name, model_config in python_config['models'].items():
            rust_config['models'][model_name.lower()] = convert_model_config(model_config)
    
    # Training configuration
    if 'training' in python_config:
        training_config = python_config['training']
        rust_config['training'] = {
            'max_steps': training_config.get('max_steps', 1000),
            'batch_size': training_config.get('batch_size', 32),
            'learning_rate': training_config.get('learning_rate', 0.001),
        }
        
        if 'early_stopping' in training_config:
            es_config = training_config['early_stopping']
            rust_config['training']['early_stopping_patience'] = es_config.get('patience', 50)
            rust_config['training']['early_stopping_min_delta'] = es_config.get('min_delta', 0.001)
    
    # Logging configuration
    if 'logging' in python_config:
        log_config = python_config['logging']
        rust_config['logging'] = {
            'level': log_config.get('level', 'info').lower(),
        }
        if 'file' in log_config:
            rust_config['logging']['file'] = log_config['file']
    
    return rust_config

def convert_frequency(python_freq: str) -> str:
    """Convert Python frequency string to Rust enum."""
    freq_mapping = {
        'D': 'Daily',
        'H': 'Hourly',
        'M': 'Monthly',
        'W': 'Weekly',
        'Y': 'Yearly',
        'Q': 'Quarterly',
        'B': 'BusinessDaily',
        'T': 'Minutely',
        'S': 'Secondly',
    }
    return freq_mapping.get(python_freq, 'Daily')

def convert_model_config(python_model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Python model configuration to Rust format."""
    rust_config = {}
    
    # Parameter name mapping
    param_mapping = {
        'h': 'horizon',
        'input_size': 'input_size',
        'hidden_size': 'hidden_size',
        'num_layers': 'num_layers',
        'learning_rate': 'learning_rate',
        'max_steps': 'max_steps',
        'batch_size': 'batch_size',
        'dropout': 'dropout',
        'random_state': 'random_seed',
        'early_stop_patience_steps': 'early_stopping_patience',
        'val_check_steps': 'validation_check_steps',
    }
    
    for python_param, rust_param in param_mapping.items():
        if python_param in python_model_config:
            rust_config[rust_param] = python_model_config[python_param]
    
    return rust_config

def convert_config_file(input_file: str, output_file: str = None):
    """Convert configuration file from Python format to Rust TOML."""
    input_path = Path(input_file)
    
    # Determine input format
    if input_path.suffix == '.yaml' or input_path.suffix == '.yml':
        with open(input_path, 'r') as f:
            python_config = yaml.safe_load(f)
    elif input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            python_config = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Convert to Rust format
    rust_config = convert_python_config_to_rust(python_config)
    
    # Write TOML output
    if output_file is None:
        output_file = input_path.stem + '_rust.toml'
    
    with open(output_file, 'w') as f:
        toml.dump(rust_config, f)
    
    print(f"Converted {input_file} to {output_file}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_config.py <input_config> [output_config]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_config_file(input_file, output_file)
```

### Data Format Converter

**Script: `convert_data.py`**
```python
#!/usr/bin/env python3
"""
Convert pandas-based data files to polars-compatible format.
"""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Optional

def convert_pandas_to_polars(input_file: str, output_file: Optional[str] = None, format: str = 'parquet'):
    """Convert pandas-readable file to polars-optimized format."""
    input_path = Path(input_file)
    
    # Read with pandas
    if input_path.suffix == '.csv':
        df_pandas = pd.read_csv(input_file)
    elif input_path.suffix == '.parquet':
        df_pandas = pd.read_parquet(input_file)
    elif input_path.suffix in ['.xlsx', '.xls']:
        df_pandas = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    
    # Clean and standardize data
    df_pandas = clean_neuralforecast_data(df_pandas)
    
    # Convert to polars
    df_polars = pl.from_pandas(df_pandas)
    
    # Optimize schema
    df_polars = optimize_schema(df_polars)
    
    # Write output
    if output_file is None:
        output_file = f"{input_path.stem}_polars.{format}"
    
    if format == 'parquet':
        df_polars.write_parquet(output_file)
    elif format == 'csv':
        df_polars.write_csv(output_file)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    print(f"Converted {input_file} to {output_file}")
    print(f"Original shape: {df_pandas.shape}")
    print(f"Converted shape: {df_polars.shape}")
    print(f"Schema: {df_polars.schema}")

def clean_neuralforecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize data for NeuralForecast compatibility."""
    df = df.copy()
    
    # Ensure required columns exist
    required_cols = ['unique_id', 'ds', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean data types
    df['unique_id'] = df['unique_id'].astype(str)
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Remove rows with null target values
    df = df.dropna(subset=['y'])
    
    # Sort by unique_id and ds
    df = df.sort_values(['unique_id', 'ds'])
    
    return df

def optimize_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Optimize polars DataFrame schema for performance."""
    # Optimize data types
    optimizations = []
    
    for col_name, dtype in df.schema.items():
        if col_name == 'unique_id':
            # Use categorical for repeated string values
            optimizations.append(pl.col(col_name).cast(pl.Categorical))
        elif col_name == 'y' and dtype == pl.Float64:
            # Use Float32 if precision allows
            optimizations.append(pl.col(col_name).cast(pl.Float32))
        else:
            optimizations.append(pl.col(col_name))
    
    return df.select(optimizations)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_data.py <input_file> [output_file] [format]")
        print("Formats: parquet (default), csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    format = sys.argv[3] if len(sys.argv) > 3 else 'parquet'
    
    convert_pandas_to_polars(input_file, output_file, format)
```

## Validation and Testing Tools

### Accuracy Validator

**Script: `validate_accuracy.py`**
```python
#!/usr/bin/env python3
"""
Validate that Rust implementation produces equivalent results to Python.
"""

import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
from typing import Dict, Tuple

def run_python_neuralforecast(data_file: str, config_file: str) -> pd.DataFrame:
    """Run Python NeuralForecast and return predictions."""
    # This would call your existing Python code
    # For demonstration, we'll simulate the process
    
    script = f"""
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
import json

# Load data and config
df = pd.read_csv('{data_file}')
with open('{config_file}', 'r') as f:
    config = json.load(f)

# Train model
model_config = config['models']['LSTM']
model = LSTM(**model_config)
nf = NeuralForecast(models=[model], freq=config['data']['freq'])
nf.fit(df)

# Generate predictions
forecasts = nf.predict()
forecasts.to_csv('python_predictions.csv', index=False)
"""
    
    with open('temp_python_script.py', 'w') as f:
        f.write(script)
    
    result = subprocess.run(['python', 'temp_python_script.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Python script failed: {result.stderr}")
    
    return pd.read_csv('python_predictions.csv')

def run_rust_neuro_divergent(data_file: str, config_file: str) -> pd.DataFrame:
    """Run Rust neuro-divergent and return predictions."""
    # Convert config to TOML format
    subprocess.run(['python', 'convert_config.py', config_file, 'rust_config.toml'])
    
    # Run Rust binary
    result = subprocess.run([
        './target/release/neuro-divergent',
        'predict',
        '--data', data_file,
        '--config', 'rust_config.toml',
        '--output', 'rust_predictions.csv'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Rust binary failed: {result.stderr}")
    
    return pd.read_csv('rust_predictions.csv')

def compare_predictions(python_df: pd.DataFrame, rust_df: pd.DataFrame, tolerance: float = 1e-6) -> Dict:
    """Compare Python and Rust predictions."""
    # Align dataframes
    merged = python_df.merge(rust_df, on=['unique_id', 'ds'], suffixes=('_python', '_rust'))
    
    # Calculate metrics
    python_pred = merged.iloc[:, -2]  # Assuming last two columns are predictions
    rust_pred = merged.iloc[:, -1]
    
    mae = np.mean(np.abs(python_pred - rust_pred))
    mse = np.mean((python_pred - rust_pred) ** 2)
    max_error = np.max(np.abs(python_pred - rust_pred))
    
    # Calculate relative errors
    relative_errors = np.abs((python_pred - rust_pred) / (python_pred + 1e-8))
    mean_relative_error = np.mean(relative_errors)
    max_relative_error = np.max(relative_errors)
    
    # Check if within tolerance
    within_tolerance = mae < tolerance
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'max_error': float(max_error),
        'mean_relative_error': float(mean_relative_error),
        'max_relative_error': float(max_relative_error),
        'within_tolerance': within_tolerance,
        'tolerance': tolerance,
        'num_predictions': len(merged)
    }

def generate_validation_report(comparison: Dict, output_file: str = 'validation_report.json'):
    """Generate detailed validation report."""
    report = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'accuracy_metrics': comparison,
        'status': 'PASS' if comparison['within_tolerance'] else 'FAIL',
        'recommendations': []
    }
    
    # Add recommendations based on results
    if not comparison['within_tolerance']:
        if comparison['max_error'] > 1.0:
            report['recommendations'].append(
                "Large prediction differences detected. Check model parameters and data preprocessing."
            )
        if comparison['mean_relative_error'] > 0.1:
            report['recommendations'].append(
                "High relative errors detected. Verify numerical precision and calculation methods."
            )
    else:
        report['recommendations'].append(
            "Validation passed. Rust implementation produces equivalent results."
        )
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Validation report saved to {output_file}")
    return report

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python validate_accuracy.py <data_file> <config_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    config_file = sys.argv[2]
    
    print("Running Python NeuralForecast...")
    python_predictions = run_python_neuralforecast(data_file, config_file)
    
    print("Running Rust neuro-divergent...")
    rust_predictions = run_rust_neuro_divergent(data_file, config_file)
    
    print("Comparing predictions...")
    comparison = compare_predictions(python_predictions, rust_predictions)
    
    print("Generating validation report...")
    report = generate_validation_report(comparison)
    
    print(f"\nValidation Results:")
    print(f"Status: {report['status']}")
    print(f"MAE: {comparison['mae']:.6f}")
    print(f"Max Error: {comparison['max_error']:.6f}")
    print(f"Mean Relative Error: {comparison['mean_relative_error']:.6f}")
```

## Performance Benchmarking

### Benchmark Runner

**Script: `run_benchmarks.py`**
```python
#!/usr/bin/env python3
"""
Run comprehensive performance benchmarks comparing Python and Rust.
"""

import time
import psutil
import subprocess
import json
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkResult:
    name: str
    python_time: float
    rust_time: float
    python_memory: float
    rust_memory: float
    speedup: float
    memory_reduction: float

def measure_python_performance(script: str, data_file: str) -> Dict[str, float]:
    """Measure Python script performance."""
    process = psutil.Popen(['python', script, data_file])
    
    start_time = time.time()
    max_memory = 0
    
    while process.poll() is None:
        try:
            memory_info = process.memory_info()
            max_memory = max(max_memory, memory_info.rss / 1024 / 1024)  # MB
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        'time': execution_time,
        'memory': max_memory
    }

def measure_rust_performance(binary: str, args: List[str]) -> Dict[str, float]:
    """Measure Rust binary performance."""
    process = psutil.Popen([binary] + args)
    
    start_time = time.time()
    max_memory = 0
    
    while process.poll() is None:
        try:
            memory_info = process.memory_info()
            max_memory = max(max_memory, memory_info.rss / 1024 / 1024)  # MB
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        'time': execution_time,
        'memory': max_memory
    }

def run_training_benchmark(data_file: str, config_file: str) -> BenchmarkResult:
    """Benchmark training performance."""
    # Python training
    python_perf = measure_python_performance('benchmark_python_training.py', data_file)
    
    # Rust training
    rust_perf = measure_rust_performance('./target/release/neuro-divergent', [
        'train', '--data', data_file, '--config', 'rust_config.toml'
    ])
    
    speedup = python_perf['time'] / rust_perf['time']
    memory_reduction = (python_perf['memory'] - rust_perf['memory']) / python_perf['memory'] * 100
    
    return BenchmarkResult(
        name='training',
        python_time=python_perf['time'],
        rust_time=rust_perf['time'],
        python_memory=python_perf['memory'],
        rust_memory=rust_perf['memory'],
        speedup=speedup,
        memory_reduction=memory_reduction
    )

def run_inference_benchmark(data_file: str, model_file: str) -> BenchmarkResult:
    """Benchmark inference performance."""
    # Python inference
    python_perf = measure_python_performance('benchmark_python_inference.py', data_file)
    
    # Rust inference
    rust_perf = measure_rust_performance('./target/release/neuro-divergent', [
        'predict', '--data', data_file, '--model', model_file
    ])
    
    speedup = python_perf['time'] / rust_perf['time']
    memory_reduction = (python_perf['memory'] - rust_perf['memory']) / python_perf['memory'] * 100
    
    return BenchmarkResult(
        name='inference',
        python_time=python_perf['time'],
        rust_time=rust_perf['time'],
        python_memory=python_perf['memory'],
        rust_memory=rust_perf['memory'],
        speedup=speedup,
        memory_reduction=memory_reduction
    )

def generate_benchmark_report(results: List[BenchmarkResult], output_file: str = 'benchmark_report.json'):
    """Generate comprehensive benchmark report."""
    report = {
        'benchmark_timestamp': pd.Timestamp.now().isoformat(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'platform': psutil.platform.platform()
        },
        'results': [asdict(result) for result in results],
        'summary': {
            'average_speedup': sum(r.speedup for r in results) / len(results),
            'average_memory_reduction': sum(r.memory_reduction for r in results) / len(results),
            'best_speedup': max(r.speedup for r in results),
            'best_memory_reduction': max(r.memory_reduction for r in results)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print(f"Average Speedup: {report['summary']['average_speedup']:.2f}x")
    print(f"Average Memory Reduction: {report['summary']['average_memory_reduction']:.1f}%")
    print(f"Best Speedup: {report['summary']['best_speedup']:.2f}x")
    print(f"Best Memory Reduction: {report['summary']['best_memory_reduction']:.1f}%")
    
    return report

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python run_benchmarks.py <data_file> <config_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    config_file = sys.argv[2]
    
    print("Running performance benchmarks...")
    
    results = []
    
    print("Benchmarking training performance...")
    training_result = run_training_benchmark(data_file, config_file)
    results.append(training_result)
    
    print("Benchmarking inference performance...")
    inference_result = run_inference_benchmark(data_file, 'model.bin')
    results.append(inference_result)
    
    print("Generating benchmark report...")
    report = generate_benchmark_report(results)
```

## CI/CD Integration

### GitHub Actions Workflow

**File: `.github/workflows/migration-validation.yml`**
```yaml
name: Migration Validation

on:
  push:
    branches: [main, migration/*]
  pull_request:
    branches: [main]

jobs:
  validate-migration:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Set up Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    
    - name: Install Python dependencies
      run: |
        pip install neuralforecast pandas polars pyyaml
    
    - name: Build Rust project
      run: |
        cargo build --release
    
    - name: Run migration analysis
      run: |
        python scripts/analyze_python_code.py ./examples/python/
    
    - name: Convert test configurations
      run: |
        python scripts/convert_config.py ./examples/config.yaml
    
    - name: Validate accuracy
      run: |
        python scripts/validate_accuracy.py ./data/test_data.csv ./examples/config.yaml
    
    - name: Run benchmarks
      run: |
        python scripts/run_benchmarks.py ./data/test_data.csv ./examples/config.yaml
    
    - name: Upload reports
      uses: actions/upload-artifact@v3
      with:
        name: migration-reports
        path: |
          migration_analysis.json
          validation_report.json
          benchmark_report.json
    
    - name: Comment on PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          
          // Read validation report
          const validation = JSON.parse(fs.readFileSync('validation_report.json', 'utf8'));
          const benchmark = JSON.parse(fs.readFileSync('benchmark_report.json', 'utf8'));
          
          const comment = `
          ## Migration Validation Results
          
          **Accuracy Validation:** ${validation.status}
          - MAE: ${validation.accuracy_metrics.mae.toFixed(6)}
          - Max Error: ${validation.accuracy_metrics.max_error.toFixed(6)}
          
          **Performance Benchmarks:**
          - Average Speedup: ${benchmark.summary.average_speedup.toFixed(2)}x
          - Memory Reduction: ${benchmark.summary.average_memory_reduction.toFixed(1)}%
          
          **Recommendations:**
          ${validation.recommendations.map(r => `- ${r}`).join('\n')}
          `;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

---

**Automation Toolkit Summary**:
- ✅ **Code Analysis**: Automated Python codebase analysis for migration planning
- ✅ **Configuration Conversion**: Automatic config file conversion (YAML/JSON → TOML)
- ✅ **Data Format Conversion**: pandas to polars data pipeline conversion
- ✅ **Accuracy Validation**: Automated accuracy comparison between Python and Rust
- ✅ **Performance Benchmarking**: Comprehensive performance measurement tools
- ✅ **CI/CD Integration**: Automated validation in development workflows
- ✅ **Reporting**: Detailed migration progress and validation reports
- ✅ **Dependency Mapping**: Automatic Python-to-Rust dependency suggestions

These tools significantly reduce manual migration effort and ensure successful, validated transitions from Python NeuralForecast to Rust neuro-divergent.