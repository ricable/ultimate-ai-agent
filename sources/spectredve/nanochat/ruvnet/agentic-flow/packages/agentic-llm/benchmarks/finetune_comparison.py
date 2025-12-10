#!/usr/bin/env python3
"""
Before/After Fine-Tuning Comparison Benchmarks
Measures improvement from MCP tool fine-tuning
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))
from validation.mcp_validator import MCPToolValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineTuneComparison:
    """Compare baseline vs fine-tuned model"""

    def __init__(self, output_dir: str = "/app/benchmarks/finetune_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_test_cases(self) -> List[Dict]:
        """Get comprehensive test cases for MCP tools"""
        return [
            # READ tool
            {"query": "Read the package.json file", "expected_tool": "read", "expected_params": {"file_path": "package.json"}},
            {"query": "Show me the README", "expected_tool": "read", "expected_params": {"file_path": "README.md"}},
            {"query": "Display the config file", "expected_tool": "read", "expected_params": {}},

            # WRITE tool
            {"query": "Create a new utils.ts file", "expected_tool": "write", "expected_params": {"file_path": "utils.ts"}},
            {"query": "Make a server.js with Express", "expected_tool": "write", "expected_params": {"file_path": "server.js"}},

            # EDIT tool
            {"query": "Change the port from 3000 to 8080", "expected_tool": "edit", "expected_params": {}},
            {"query": "Rename getData to fetchData", "expected_tool": "edit", "expected_params": {}},

            # BASH tool
            {"query": "Run the tests", "expected_tool": "bash", "expected_params": {"command": "npm test"}},
            {"query": "Install dependencies", "expected_tool": "bash", "expected_params": {"command": "npm install"}},
            {"query": "Build the project", "expected_tool": "bash", "expected_params": {}},
            {"query": "Start the development server", "expected_tool": "bash", "expected_params": {}},

            # GREP tool
            {"query": "Search for TODO comments", "expected_tool": "grep", "expected_params": {"pattern": "TODO"}},
            {"query": "Find all authentication functions", "expected_tool": "grep", "expected_params": {}},
            {"query": "Look for error handling code", "expected_tool": "grep", "expected_params": {}},

            # GLOB tool
            {"query": "List all TypeScript files", "expected_tool": "glob", "expected_params": {"pattern": "**/*.ts"}},
            {"query": "Show all test files", "expected_tool": "glob", "expected_params": {}},
            {"query": "Find all components", "expected_tool": "glob", "expected_params": {}},

            # Complex queries
            {"query": "What's in the authentication config", "expected_tool": "read", "expected_params": {}},
            {"query": "Execute the linter", "expected_tool": "bash", "expected_params": {}},
            {"query": "Locate API endpoint definitions", "expected_tool": "grep", "expected_params": {}},
            {"query": "Get all JavaScript files in src", "expected_tool": "glob", "expected_params": {}},
        ]

    def benchmark_model(
        self,
        model_path: str,
        model_name: str,
        test_cases: List[Dict]
    ) -> Dict:
        """Benchmark a model on MCP tool calling"""
        logger.info(f"\nBenchmarking: {model_name}")
        logger.info("=" * 60)

        try:
            # Create validator
            validator = MCPToolValidator(model_path=model_path)

            # Run validation
            start_time = time.time()
            results = validator.validate_tool_accuracy(test_cases)
            duration = time.time() - start_time

            # Add metadata
            results["model_name"] = model_name
            results["model_path"] = model_path
            results["duration_seconds"] = duration
            results["avg_time_per_test"] = duration / len(test_cases)

            return results

        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
                "total": len(test_cases),
                "accuracy": 0.0
            }

    def calculate_improvements(
        self,
        baseline: Dict,
        finetuned: Dict
    ) -> Dict:
        """Calculate improvement metrics"""
        improvements = {
            "accuracy_improvement": finetuned["accuracy"] - baseline["accuracy"],
            "accuracy_improvement_pct": (
                (finetuned["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
                if baseline["accuracy"] > 0 else 0
            ),
            "param_accuracy_improvement": finetuned["param_accuracy"] - baseline["param_accuracy"],
            "tool_detection_improvement": (
                (finetuned["tool_found"] / finetuned["total"]) -
                (baseline["tool_found"] / baseline["total"])
            ) * 100,
            "meets_target": finetuned["accuracy"] >= 95.0,
            "quality_tier": self._get_quality_tier(finetuned["accuracy"])
        }

        return improvements

    def _get_quality_tier(self, accuracy: float) -> str:
        """Get quality tier based on accuracy"""
        if accuracy >= 95:
            return "EXCELLENT - Production Ready"
        elif accuracy >= 85:
            return "GOOD - Acceptable for most use cases"
        elif accuracy >= 75:
            return "FAIR - Needs improvement"
        else:
            return "POOR - Significant training needed"

    def run_comparison(
        self,
        baseline_path: str = None,
        finetuned_path: str = "/app/checkpoints/mcp_finetuned/final_model"
    ) -> Dict:
        """Run complete before/after comparison"""
        logger.info("\n" + "=" * 60)
        logger.info("MCP FINE-TUNING COMPARISON BENCHMARK")
        logger.info("=" * 60)

        # Get test cases
        test_cases = self.get_test_cases()
        logger.info(f"\nTest cases: {len(test_cases)}")

        # Benchmark baseline (if provided)
        baseline_results = None
        if baseline_path:
            baseline_results = self.benchmark_model(
                model_path=baseline_path,
                model_name="Baseline (Pre-fine-tuning)",
                test_cases=test_cases
            )

        # Benchmark fine-tuned model
        finetuned_results = self.benchmark_model(
            model_path=finetuned_path,
            model_name="Fine-tuned (MCP Tools)",
            test_cases=test_cases
        )

        # Calculate improvements
        improvements = None
        if baseline_results:
            improvements = self.calculate_improvements(baseline_results, finetuned_results)

        # Create report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_cases": len(test_cases),
            "baseline": baseline_results,
            "finetuned": finetuned_results,
            "improvements": improvements
        }

        # Print summary
        self._print_summary(report)

        # Save report
        report_path = self.output_dir / "finetune_comparison.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n📄 Report saved to: {report_path}")

        return report

    def _print_summary(self, report: Dict):
        """Print comparison summary"""
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)

        baseline = report.get("baseline")
        finetuned = report["finetuned"]
        improvements = report.get("improvements")

        # Fine-tuned results
        logger.info(f"\n🎯 Fine-tuned Model:")
        logger.info(f"  Tool Accuracy: {finetuned['accuracy']:.1f}%")
        logger.info(f"  Parameter Accuracy: {finetuned['param_accuracy']:.1f}%")
        logger.info(f"  Tool Detection: {finetuned['tool_found']}/{finetuned['total']}")
        logger.info(f"  Quality Tier: {self._get_quality_tier(finetuned['accuracy'])}")

        # Improvements
        if improvements:
            logger.info(f"\n📈 Improvements:")
            logger.info(f"  Accuracy Gain: +{improvements['accuracy_improvement']:.1f}% ({improvements['accuracy_improvement_pct']:.1f}% improvement)")
            logger.info(f"  Tool Detection: +{improvements['tool_detection_improvement']:.1f}%")
            logger.info(f"  Meets 95% Target: {'✅ YES' if improvements['meets_target'] else '❌ NO'}")

        # Recommendation
        logger.info(f"\n💡 Recommendation:")
        if finetuned['accuracy'] >= 95:
            logger.info("  ✅ Model is production-ready for Claude Agent SDK")
            logger.info("  ✅ Proceed with ONNX quantization and deployment")
        elif finetuned['accuracy'] >= 85:
            logger.info("  ✓ Model performs well for most use cases")
            logger.info("  → Consider additional fine-tuning for critical applications")
        else:
            logger.info("  ⚠ Model needs improvement")
            logger.info("  → Recommended: More training data or additional epochs")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned model")
    parser.add_argument("--baseline", type=str, help="Path to baseline model (optional)")
    parser.add_argument("--finetuned", type=str, default="/app/checkpoints/mcp_finetuned/final_model", help="Path to fine-tuned model")
    parser.add_argument("--output-dir", type=str, default="/app/benchmarks/finetune_results", help="Output directory")

    args = parser.parse_args()

    # Run comparison
    comparison = FineTuneComparison(output_dir=args.output_dir)
    report = comparison.run_comparison(
        baseline_path=args.baseline,
        finetuned_path=args.finetuned
    )

    logger.info("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
