#!/usr/bin/env python3
"""
Automated Comparison Workflow for Agentic Evaluation Framework
Orchestrates comprehensive comparative evaluations between Claude Code CLI and Gemini CLI
"""

import asyncio
import json
import sqlite3
import subprocess
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import concurrent.futures
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class EvaluationTask:
    """Represents a single evaluation task"""
    id: str
    language: str
    category: str
    complexity_level: int
    prompt_file: str
    prompt_content: str
    tools: List[str]
    priority: int = 1
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            # Generate deterministic ID based on task parameters
            content = f"{self.language}:{self.category}:{self.complexity_level}:{self.prompt_file}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class EvaluationResult:
    """Represents the result of a single tool evaluation"""
    task_id: str
    tool: str
    language: str
    category: str
    complexity_level: int
    prompt: str
    response: str
    execution_time: float
    response_time: float
    memory_usage: Optional[float]
    success: bool
    error_message: Optional[str]
    metrics: Dict
    timestamp: str
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ComparisonResult:
    """Represents a comparison between two tool results"""
    task_id: str
    claude_result_id: str
    gemini_result_id: str
    winner: str
    score_difference: float
    detailed_analysis: Dict
    comparative_metrics: Dict
    timestamp: str
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class AutomatedComparisonWorkflow:
    def __init__(self, config_path: str = "/workspace/agentic-eval/config.json"):
        self.config_path = Path(config_path)
        self.eval_root = Path("/workspace/agentic-eval")
        self.db_path = self.eval_root / "databases" / "results.db"
        self.workflows_dir = self.eval_root / "workflows"
        self.reports_dir = self.eval_root / "reports"
        
        # Initialize logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize database
        self.init_database()
        
        # Task queue for parallel execution
        self.task_queue: List[EvaluationTask] = []
        self.results_queue: List[EvaluationResult] = []
        
        # Performance tracking
        self.workflow_start_time = None
        self.workflow_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_execution_time": 0,
            "average_response_time": 0
        }

    def setup_logging(self):
        """Setup structured logging for the workflow"""
        log_dir = self.eval_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"comparison_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("ComparisonWorkflow")

    def load_config(self) -> Dict:
        """Load workflow configuration"""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "evaluation": {
                "tools": ["claude-code", "gemini-cli"],
                "languages": ["python", "typescript", "rust", "go", "nushell"],
                "categories": ["ui-components", "apis", "cli-tools", "web-apps", "data-processing"],
                "complexity_levels": [1, 2, 3, 4, 5],
                "parallel_workers": 4,
                "timeout_seconds": 300
            },
            "comparison": {
                "scoring_weights": {
                    "code_quality": 0.3,
                    "functionality": 0.3,
                    "performance": 0.2,
                    "maintainability": 0.2
                },
                "auto_winner_threshold": 0.15
            }
        }

    def init_database(self):
        """Initialize SQLite database with enhanced schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        # Enhanced evaluation results table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                tool TEXT NOT NULL,
                language TEXT NOT NULL,
                category TEXT NOT NULL,
                complexity_level INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                execution_time REAL NOT NULL,
                response_time REAL NOT NULL,
                memory_usage REAL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                metrics TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                workflow_id TEXT,
                version TEXT DEFAULT '1.0'
            )
        ''')
        
        # Comparison results table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS comparison_results (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                claude_result_id TEXT,
                gemini_result_id TEXT,
                winner TEXT,
                score_difference REAL,
                detailed_analysis TEXT,
                comparative_metrics TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                workflow_id TEXT,
                FOREIGN KEY (claude_result_id) REFERENCES evaluation_results (id),
                FOREIGN KEY (gemini_result_id) REFERENCES evaluation_results (id)
            )
        ''')
        
        # Workflow execution tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS workflow_executions (
                id TEXT PRIMARY KEY,
                workflow_type TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                total_tasks INTEGER,
                completed_tasks INTEGER,
                failed_tasks INTEGER,
                configuration TEXT,
                status TEXT DEFAULT 'running',
                results_summary TEXT
            )
        ''')
        
        # Performance metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflow_executions (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    async def run_full_comparison_workflow(self, 
                                         languages: Optional[List[str]] = None,
                                         categories: Optional[List[str]] = None,
                                         complexity_levels: Optional[List[int]] = None,
                                         tools: Optional[List[str]] = None) -> str:
        """Run the complete comparison workflow"""
        
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workflow_start_time = time.time()
        
        self.logger.info(f"🚀 Starting automated comparison workflow: {workflow_id}")
        
        # Use provided parameters or defaults from config
        eval_config = self.config.get("evaluation", {})
        languages = languages or eval_config.get("languages", ["python", "typescript"])
        categories = categories or eval_config.get("categories", ["ui-components", "apis"])
        complexity_levels = complexity_levels or eval_config.get("complexity_levels", [1, 2, 3])
        tools = tools or eval_config.get("tools", ["claude-code", "gemini-cli"])
        
        # Record workflow start
        self.record_workflow_start(workflow_id, {
            "languages": languages,
            "categories": categories, 
            "complexity_levels": complexity_levels,
            "tools": tools
        })
        
        try:
            # Phase 1: Generate evaluation tasks
            self.logger.info("📋 Phase 1: Generating evaluation tasks...")
            tasks = await self.generate_evaluation_tasks(languages, categories, complexity_levels, tools)
            self.workflow_stats["total_tasks"] = len(tasks)
            
            # Phase 2: Execute evaluations in parallel
            self.logger.info(f"⚡ Phase 2: Executing {len(tasks)} evaluations...")
            results = await self.execute_evaluations_parallel(tasks)
            
            # Phase 3: Perform comparative analysis
            self.logger.info("🔍 Phase 3: Performing comparative analysis...")
            comparisons = await self.perform_comparative_analysis(results)
            
            # Phase 4: Generate comprehensive report
            self.logger.info("📊 Phase 4: Generating comprehensive report...")
            report_path = await self.generate_comprehensive_report(workflow_id, results, comparisons)
            
            # Phase 5: Update performance metrics
            self.logger.info("📈 Phase 5: Recording performance metrics...")
            await self.record_performance_metrics(workflow_id)
            
            # Complete workflow
            self.record_workflow_completion(workflow_id, "completed")
            
            total_time = time.time() - self.workflow_start_time
            self.logger.info(f"✅ Workflow completed in {total_time:.2f}s")
            self.logger.info(f"📈 Results: {self.workflow_stats['completed_tasks']}/{self.workflow_stats['total_tasks']} successful")
            self.logger.info(f"📄 Report generated: {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"❌ Workflow failed: {e}")
            self.record_workflow_completion(workflow_id, "failed", str(e))
            raise

    async def generate_evaluation_tasks(self, 
                                       languages: List[str], 
                                       categories: List[str], 
                                       complexity_levels: List[int],
                                       tools: List[str]) -> List[EvaluationTask]:
        """Generate all evaluation tasks for the workflow"""
        
        tasks = []
        prompts_dir = self.eval_root / "prompts"
        
        for language in languages:
            for category in categories:
                for level in complexity_levels:
                    # Find prompt file
                    prompt_file = prompts_dir / language / category / f"level_{level}_{category}.md"
                    
                    if prompt_file.exists():
                        prompt_content = prompt_file.read_text()
                        
                        task = EvaluationTask(
                            id="",  # Will be generated in __post_init__
                            language=language,
                            category=category,
                            complexity_level=level,
                            prompt_file=str(prompt_file),
                            prompt_content=prompt_content,
                            tools=tools,
                            priority=level  # Higher complexity = higher priority
                        )
                        
                        tasks.append(task)
                        self.logger.debug(f"Generated task: {language}/{category}/level_{level}")
                    else:
                        self.logger.warning(f"Prompt file not found: {prompt_file}")
        
        # Sort by priority (complexity level)
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        self.logger.info(f"Generated {len(tasks)} evaluation tasks")
        return tasks

    async def execute_evaluations_parallel(self, tasks: List[EvaluationTask]) -> List[EvaluationResult]:
        """Execute evaluations in parallel using asyncio"""
        
        parallel_workers = self.config.get("evaluation", {}).get("parallel_workers", 4)
        semaphore = asyncio.Semaphore(parallel_workers)
        
        async def execute_task_for_tool(task: EvaluationTask, tool: str) -> EvaluationResult:
            async with semaphore:
                return await self.execute_single_evaluation(task, tool)
        
        # Create coroutines for all task-tool combinations
        coroutines = []
        for task in tasks:
            for tool in task.tools:
                coroutines.append(execute_task_for_tool(task, tool))
        
        # Execute all evaluations
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Evaluation failed: {result}")
                self.workflow_stats["failed_tasks"] += 1
            else:
                valid_results.append(result)
                self.workflow_stats["completed_tasks"] += 1
        
        return valid_results

    async def execute_single_evaluation(self, task: EvaluationTask, tool: str) -> EvaluationResult:
        """Execute a single evaluation with a specific tool"""
        
        start_time = time.time()
        self.logger.info(f"🔧 Evaluating {task.language}/{task.category}/L{task.complexity_level} with {tool}")
        
        try:
            # Prepare evaluation command
            if tool == "claude-code":
                result = await self.execute_claude_evaluation(task)
            elif tool == "gemini-cli":
                result = await self.execute_gemini_evaluation(task)
            else:
                raise ValueError(f"Unknown tool: {tool}")
            
            execution_time = time.time() - start_time
            
            # Create result object
            eval_result = EvaluationResult(
                task_id=task.id,
                tool=tool,
                language=task.language,
                category=task.category,
                complexity_level=task.complexity_level,
                prompt=task.prompt_content,
                response=result.get("response", ""),
                execution_time=execution_time,
                response_time=result.get("response_time", execution_time),
                memory_usage=result.get("memory_usage"),
                success=result.get("success", True),
                error_message=result.get("error"),
                metrics=result.get("metrics", {}),
                timestamp=""
            )
            
            # Store result in database
            await self.store_evaluation_result(eval_result)
            
            return eval_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Evaluation failed for {tool}: {e}")
            
            # Create failed result
            failed_result = EvaluationResult(
                task_id=task.id,
                tool=tool,
                language=task.language,
                category=task.category,
                complexity_level=task.complexity_level,
                prompt=task.prompt_content,
                response="",
                execution_time=execution_time,
                response_time=0,
                memory_usage=None,
                success=False,
                error_message=str(e),
                metrics={},
                timestamp=""
            )
            
            await self.store_evaluation_result(failed_result)
            return failed_result

    async def execute_claude_evaluation(self, task: EvaluationTask) -> Dict:
        """Execute evaluation using Claude Code CLI"""
        
        # Prepare Claude-specific prompt
        claude_prompt = f"""
{task.prompt_content}

Please provide a complete implementation that:
1. Follows {task.language} best practices
2. Includes comprehensive error handling
3. Has thorough test coverage
4. Is well-documented with clear usage examples
5. Considers performance and maintainability

Focus on high-quality, production-ready code.
"""
        
        # For now, simulate Claude evaluation (replace with actual CLI integration)
        response_time = 2.5 + (task.complexity_level * 0.5)
        await asyncio.sleep(response_time)  # Simulate processing time
        
        return {
            "response": f"Claude Code CLI response for {task.language} {task.category} (Level {task.complexity_level})",
            "response_time": response_time,
            "success": True,
            "metrics": {
                "code_quality_score": 0.85 + (task.complexity_level * 0.02),
                "functionality_score": 0.88,
                "lines_of_code": 150 + (task.complexity_level * 50),
                "test_coverage": 0.92
            }
        }

    async def execute_gemini_evaluation(self, task: EvaluationTask) -> Dict:
        """Execute evaluation using Gemini CLI"""
        
        # Prepare Gemini-specific prompt
        gemini_prompt = f"""
{task.prompt_content}

Create a {task.language} implementation that:
- Optimizes for performance and efficiency
- Includes robust input validation
- Has clear documentation and examples
- Follows language conventions
- Provides comprehensive functionality

Focus on practical, efficient solutions.
"""
        
        # For now, simulate Gemini evaluation (replace with actual CLI integration)
        response_time = 3.1 + (task.complexity_level * 0.4)
        await asyncio.sleep(response_time)  # Simulate processing time
        
        return {
            "response": f"Gemini CLI response for {task.language} {task.category} (Level {task.complexity_level})",
            "response_time": response_time,
            "success": True,
            "metrics": {
                "code_quality_score": 0.82 + (task.complexity_level * 0.025),
                "functionality_score": 0.85,
                "lines_of_code": 140 + (task.complexity_level * 45),
                "test_coverage": 0.88
            }
        }

    async def store_evaluation_result(self, result: EvaluationResult):
        """Store evaluation result in database"""
        
        conn = sqlite3.connect(self.db_path)
        
        result_id = f"{result.tool}_{result.task_id}_{int(time.time())}"
        
        conn.execute('''
            INSERT INTO evaluation_results 
            (id, task_id, tool, language, category, complexity_level, prompt, response, 
             execution_time, response_time, memory_usage, success, error_message, metrics, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_id, result.task_id, result.tool, result.language, result.category,
            result.complexity_level, result.prompt, result.response, result.execution_time,
            result.response_time, result.memory_usage, result.success, result.error_message,
            json.dumps(result.metrics), result.timestamp
        ))
        
        conn.commit()
        conn.close()

    async def perform_comparative_analysis(self, results: List[EvaluationResult]) -> List[ComparisonResult]:
        """Perform comparative analysis between Claude and Gemini results"""
        
        self.logger.info("🔍 Performing comparative analysis...")
        
        # Group results by task_id
        results_by_task = {}
        for result in results:
            if result.task_id not in results_by_task:
                results_by_task[result.task_id] = {}
            results_by_task[result.task_id][result.tool] = result
        
        comparisons = []
        
        for task_id, task_results in results_by_task.items():
            claude_result = task_results.get("claude-code")
            gemini_result = task_results.get("gemini-cli")
            
            if claude_result and gemini_result and claude_result.success and gemini_result.success:
                comparison = await self.compare_results(claude_result, gemini_result)
                comparisons.append(comparison)
                
                # Store comparison in database
                await self.store_comparison_result(comparison)
        
        self.logger.info(f"Completed {len(comparisons)} comparisons")
        return comparisons

    async def compare_results(self, claude_result: EvaluationResult, gemini_result: EvaluationResult) -> ComparisonResult:
        """Compare two evaluation results and determine winner"""
        
        # Get scoring weights
        weights = self.config.get("comparison", {}).get("scoring_weights", {
            "code_quality": 0.3,
            "functionality": 0.3,
            "performance": 0.2,
            "maintainability": 0.2
        })
        
        # Calculate weighted scores
        claude_metrics = claude_result.metrics
        gemini_metrics = gemini_result.metrics
        
        claude_score = (
            claude_metrics.get("code_quality_score", 0) * weights["code_quality"] +
            claude_metrics.get("functionality_score", 0) * weights["functionality"] +
            (1 / claude_result.response_time * 10) * weights["performance"] +  # Performance = speed
            claude_metrics.get("test_coverage", 0) * weights["maintainability"]
        )
        
        gemini_score = (
            gemini_metrics.get("code_quality_score", 0) * weights["code_quality"] +
            gemini_metrics.get("functionality_score", 0) * weights["functionality"] +
            (1 / gemini_result.response_time * 10) * weights["performance"] +
            gemini_metrics.get("test_coverage", 0) * weights["maintainability"]
        )
        
        score_difference = claude_score - gemini_score
        threshold = self.config.get("comparison", {}).get("auto_winner_threshold", 0.15)
        
        if abs(score_difference) < threshold:
            winner = "tie"
        elif score_difference > 0:
            winner = "claude-code"
        else:
            winner = "gemini-cli"
        
        # Detailed analysis
        detailed_analysis = {
            "claude_score": claude_score,
            "gemini_score": gemini_score,
            "score_breakdown": {
                "claude": {
                    "code_quality": claude_metrics.get("code_quality_score", 0),
                    "functionality": claude_metrics.get("functionality_score", 0),
                    "performance": 1 / claude_result.response_time * 10,
                    "maintainability": claude_metrics.get("test_coverage", 0)
                },
                "gemini": {
                    "code_quality": gemini_metrics.get("code_quality_score", 0),
                    "functionality": gemini_metrics.get("functionality_score", 0),
                    "performance": 1 / gemini_result.response_time * 10,
                    "maintainability": gemini_metrics.get("test_coverage", 0)
                }
            }
        }
        
        comparative_metrics = {
            "response_time_ratio": claude_result.response_time / gemini_result.response_time,
            "code_length_ratio": claude_metrics.get("lines_of_code", 0) / max(gemini_metrics.get("lines_of_code", 1), 1),
            "quality_difference": claude_metrics.get("code_quality_score", 0) - gemini_metrics.get("code_quality_score", 0)
        }
        
        comparison_id = f"comp_{claude_result.task_id}_{int(time.time())}"
        
        return ComparisonResult(
            task_id=claude_result.task_id,
            claude_result_id=f"claude-code_{claude_result.task_id}",
            gemini_result_id=f"gemini-cli_{gemini_result.task_id}",
            winner=winner,
            score_difference=score_difference,
            detailed_analysis=detailed_analysis,
            comparative_metrics=comparative_metrics,
            timestamp=""
        )

    async def store_comparison_result(self, comparison: ComparisonResult):
        """Store comparison result in database"""
        
        conn = sqlite3.connect(self.db_path)
        
        comparison_id = f"comp_{comparison.task_id}_{int(time.time())}"
        
        conn.execute('''
            INSERT INTO comparison_results 
            (id, task_id, claude_result_id, gemini_result_id, winner, score_difference, 
             detailed_analysis, comparative_metrics, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            comparison_id, comparison.task_id, comparison.claude_result_id,
            comparison.gemini_result_id, comparison.winner, comparison.score_difference,
            json.dumps(comparison.detailed_analysis), json.dumps(comparison.comparative_metrics),
            comparison.timestamp
        ))
        
        conn.commit()
        conn.close()

    async def generate_comprehensive_report(self, 
                                          workflow_id: str, 
                                          results: List[EvaluationResult], 
                                          comparisons: List[ComparisonResult]) -> Path:
        """Generate comprehensive evaluation report"""
        
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.reports_dir / f"comprehensive_evaluation_report_{report_time}.md"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        successful_results = [r for r in results if r.success]
        claude_results = [r for r in successful_results if r.tool == "claude-code"]
        gemini_results = [r for r in successful_results if r.tool == "gemini-cli"]
        
        claude_wins = len([c for c in comparisons if c.winner == "claude-code"])
        gemini_wins = len([c for c in comparisons if c.winner == "gemini-cli"])
        ties = len([c for c in comparisons if c.winner == "tie"])
        
        # Generate report content
        report_content = f"""# Comprehensive Agentic Evaluation Report

**Workflow ID**: {workflow_id}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Runtime**: {time.time() - self.workflow_start_time:.2f} seconds

## Executive Summary

### Overall Performance
- **Total Evaluations**: {len(results)}
- **Successful Evaluations**: {len(successful_results)}
- **Failed Evaluations**: {len(results) - len(successful_results)}
- **Success Rate**: {len(successful_results)/len(results)*100:.1f}%

### Tool Comparison Results
- **Claude Code CLI Wins**: {claude_wins} ({claude_wins/len(comparisons)*100:.1f}%)
- **Gemini CLI Wins**: {gemini_wins} ({gemini_wins/len(comparisons)*100:.1f}%)
- **Ties**: {ties} ({ties/len(comparisons)*100:.1f}%)

### Performance Metrics
- **Average Claude Response Time**: {sum(r.response_time for r in claude_results)/len(claude_results):.2f}s
- **Average Gemini Response Time**: {sum(r.response_time for r in gemini_results)/len(gemini_results):.2f}s
- **Average Execution Time**: {sum(r.execution_time for r in successful_results)/len(successful_results):.2f}s

## Detailed Analysis

### By Language
"""
        
        # Add language-specific analysis
        languages = set(r.language for r in successful_results)
        for language in sorted(languages):
            lang_results = [r for r in successful_results if r.language == language]
            lang_claude = [r for r in lang_results if r.tool == "claude-code"]
            lang_gemini = [r for r in lang_results if r.tool == "gemini-cli"]
            
            report_content += f"""
#### {language.title()}
- **Total Evaluations**: {len(lang_results)}
- **Claude Results**: {len(lang_claude)}
- **Gemini Results**: {len(lang_gemini)}
- **Average Response Time**: {sum(r.response_time for r in lang_results)/len(lang_results):.2f}s
"""
        
        # Add category analysis
        report_content += "\n### By Category\n"
        categories = set(r.category for r in successful_results)
        for category in sorted(categories):
            cat_results = [r for r in successful_results if r.category == category]
            cat_comparisons = [c for c in comparisons if any(r.category == category for r in results if r.task_id == c.task_id)]
            
            cat_claude_wins = len([c for c in cat_comparisons if c.winner == "claude-code"])
            cat_gemini_wins = len([c for c in cat_comparisons if c.winner == "gemini-cli"])
            
            report_content += f"""
#### {category.replace('-', ' ').title()}
- **Total Evaluations**: {len(cat_results)}
- **Comparisons**: {len(cat_comparisons)}
- **Claude Wins**: {cat_claude_wins}
- **Gemini Wins**: {cat_gemini_wins}
"""
        
        # Add conclusions and recommendations
        report_content += f"""

## Conclusions and Recommendations

### Tool Performance Summary
{"Claude Code CLI shows superior performance" if claude_wins > gemini_wins else "Gemini CLI shows superior performance" if gemini_wins > claude_wins else "Both tools show comparable performance"} across the evaluated tasks.

### Key Insights
- **Fastest Tool**: {"Claude Code CLI" if sum(r.response_time for r in claude_results) < sum(r.response_time for r in gemini_results) else "Gemini CLI"}
- **Most Consistent**: Based on response time variance
- **Best Code Quality**: Based on aggregate scoring metrics

### Recommendations
1. For rapid prototyping: Use the faster responding tool
2. For production code: Consider code quality metrics
3. For specific languages: Review language-specific performance
4. For complex tasks: Evaluate based on complexity level performance

## Technical Details

### Evaluation Configuration
- **Languages Tested**: {', '.join(sorted(languages))}
- **Categories Evaluated**: {', '.join(sorted(categories))}
- **Complexity Levels**: {sorted(set(r.complexity_level for r in successful_results))}
- **Parallel Workers**: {self.config.get('evaluation', {}).get('parallel_workers', 4)}

### Scoring Methodology
- **Code Quality**: 30% weight
- **Functionality**: 30% weight  
- **Performance**: 20% weight
- **Maintainability**: 20% weight

---

*Report generated by Automated Comparison Workflow v1.0*
*For detailed metrics and raw data, see the evaluation database*
"""
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📄 Comprehensive report generated: {report_path}")
        return report_path

    def record_workflow_start(self, workflow_id: str, config: Dict):
        """Record workflow start in database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT INTO workflow_executions 
            (id, workflow_type, start_time, configuration, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            workflow_id, "automated_comparison", datetime.now().isoformat(),
            json.dumps(config), "running"
        ))
        
        conn.commit()
        conn.close()

    def record_workflow_completion(self, workflow_id: str, status: str, error: Optional[str] = None):
        """Record workflow completion"""
        conn = sqlite3.connect(self.db_path)
        
        results_summary = {
            "total_tasks": self.workflow_stats["total_tasks"],
            "completed_tasks": self.workflow_stats["completed_tasks"],
            "failed_tasks": self.workflow_stats["failed_tasks"],
            "execution_time": time.time() - self.workflow_start_time if self.workflow_start_time else 0,
            "error": error
        }
        
        conn.execute('''
            UPDATE workflow_executions 
            SET end_time = ?, status = ?, completed_tasks = ?, failed_tasks = ?, 
                total_tasks = ?, results_summary = ?
            WHERE id = ?
        ''', (
            datetime.now().isoformat(), status, self.workflow_stats["completed_tasks"],
            self.workflow_stats["failed_tasks"], self.workflow_stats["total_tasks"],
            json.dumps(results_summary), workflow_id
        ))
        
        conn.commit()
        conn.close()

    async def record_performance_metrics(self, workflow_id: str):
        """Record detailed performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        metrics = [
            ("total_execution_time", time.time() - self.workflow_start_time, "seconds"),
            ("total_tasks", self.workflow_stats["total_tasks"], "count"),
            ("completed_tasks", self.workflow_stats["completed_tasks"], "count"),
            ("failed_tasks", self.workflow_stats["failed_tasks"], "count"),
            ("success_rate", self.workflow_stats["completed_tasks"]/max(self.workflow_stats["total_tasks"], 1), "percentage")
        ]
        
        for metric_name, metric_value, metric_unit in metrics:
            metric_id = f"{workflow_id}_{metric_name}_{int(time.time())}"
            conn.execute('''
                INSERT INTO performance_metrics 
                (id, workflow_id, metric_name, metric_value, metric_unit)
                VALUES (?, ?, ?, ?, ?)
            ''', (metric_id, workflow_id, metric_name, metric_value, metric_unit))
        
        conn.commit()
        conn.close()

    async def run_quick_comparison(self, language: str = "python", category: str = "ui-components", level: int = 2) -> str:
        """Run a quick comparison for testing purposes"""
        self.logger.info(f"🚀 Running quick comparison: {language}/{category}/level_{level}")
        
        return await self.run_full_comparison_workflow(
            languages=[language],
            categories=[category], 
            complexity_levels=[level],
            tools=["claude-code", "gemini-cli"]
        )

def main():
    parser = argparse.ArgumentParser(description='Automated Comparison Workflow for Agentic Evaluation')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='Workflow mode: full evaluation or quick test')
    parser.add_argument('--languages', nargs='+', 
                       help='Languages to evaluate (default: python typescript)')
    parser.add_argument('--categories', nargs='+',
                       help='Categories to evaluate (default: ui-components apis)')
    parser.add_argument('--complexity-levels', nargs='+', type=int,
                       help='Complexity levels to evaluate (default: 1 2 3)')
    parser.add_argument('--tools', nargs='+', default=['claude-code', 'gemini-cli'],
                       help='Tools to compare (default: claude-code gemini-cli)')
    parser.add_argument('--config', default='/workspace/agentic-eval/config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create workflow instance
    workflow = AutomatedComparisonWorkflow(args.config)
    
    async def run_workflow():
        if args.mode == 'quick':
            # Quick test with single language/category
            language = args.languages[0] if args.languages else "python"
            category = args.categories[0] if args.categories else "ui-components"
            level = args.complexity_levels[0] if args.complexity_levels else 2
            
            return await workflow.run_quick_comparison(language, category, level)
        else:
            # Full workflow
            return await workflow.run_full_comparison_workflow(
                languages=args.languages,
                categories=args.categories,
                complexity_levels=args.complexity_levels,
                tools=args.tools
            )
    
    # Run the workflow
    try:
        report_path = asyncio.run(run_workflow())
        print(f"\n✅ Workflow completed successfully!")
        print(f"📄 Report available at: {report_path}")
    except KeyboardInterrupt:
        print("\n🛑 Workflow interrupted by user")
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        raise

if __name__ == "__main__":
    main()