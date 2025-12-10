#!/usr/bin/env python3
"""
Cloud Run GPU Training Orchestrator
Manages fine-tuning pipeline with health reporting and email notifications
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.finetune_mcp import MCPFineTuner, MCPFineTuningConfig
from validation.mcp_validator import MCPValidator
from benchmarks.finetune_comparison import MCPBenchmark

import resend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CloudRunOrchestrator:
    """Orchestrate training, validation, and benchmarking on Cloud Run"""

    def __init__(self):
        self.start_time = datetime.now()
        self.status_file = Path("/app/results/training_status.json")
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Email configuration
        self.resend_api_key = os.environ.get("RESEND_API_KEY")
        self.notification_email = os.environ.get("NOTIFICATION_EMAIL")

        if self.resend_api_key:
            resend.api_key = self.resend_api_key
            logger.info(f"Email notifications enabled for: {self.notification_email}")
        else:
            logger.warning("RESEND_API_KEY not set - email notifications disabled")

    def update_status(self, phase: str, status: str, details: dict = None):
        """Update training status for monitoring"""
        status_data = {
            "phase": phase,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
            "details": details or {}
        }

        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)

        logger.info(f"Status update: {phase} - {status}")

    def run_baseline_benchmarks(self):
        """Run baseline benchmarks on original Phi-4"""
        logger.info("=" * 60)
        logger.info("PHASE 1: Baseline Benchmarking")
        logger.info("=" * 60)

        self.update_status("baseline_benchmarks", "running")

        try:
            benchmark = MCPBenchmark()
            baseline_results = benchmark.run_baseline_benchmark()

            # Save baseline results
            baseline_file = Path("/app/benchmarks/results/baseline_results.json")
            baseline_file.parent.mkdir(parents=True, exist_ok=True)

            with open(baseline_file, 'w') as f:
                json.dump(baseline_results, f, indent=2)

            self.update_status("baseline_benchmarks", "completed", baseline_results)
            logger.info(f"Baseline results saved: {baseline_file}")

            return baseline_results

        except Exception as e:
            logger.error(f"Baseline benchmark failed: {e}")
            self.update_status("baseline_benchmarks", "failed", {"error": str(e)})
            raise

    def run_finetuning(self):
        """Execute fine-tuning pipeline"""
        logger.info("=" * 60)
        logger.info("PHASE 2: Fine-Tuning")
        logger.info("=" * 60)

        self.update_status("finetuning", "running")

        try:
            config = MCPFineTuningConfig(
                output_dir="/app/checkpoints/mcp_finetuned",
                model_cache_dir="/app/models"
            )

            finetuner = MCPFineTuner(config)
            metrics = finetuner.train()

            self.update_status("finetuning", "completed", metrics)
            logger.info("Fine-tuning completed successfully")

            return metrics

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            self.update_status("finetuning", "failed", {"error": str(e)})
            raise

    def run_validation(self):
        """Validate fine-tuned model"""
        logger.info("=" * 60)
        logger.info("PHASE 3: Validation")
        logger.info("=" * 60)

        self.update_status("validation", "running")

        try:
            validator = MCPValidator(
                model_path="/app/checkpoints/mcp_finetuned/final_model",
                output_dir="/app/validation/results"
            )

            results = validator.run_validation()

            self.update_status("validation", "completed", results)
            logger.info(f"Validation results: {results}")

            return results

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.update_status("validation", "failed", {"error": str(e)})
            raise

    def run_post_training_benchmarks(self):
        """Run benchmarks on fine-tuned model"""
        logger.info("=" * 60)
        logger.info("PHASE 4: Post-Training Benchmarking")
        logger.info("=" * 60)

        self.update_status("post_training_benchmarks", "running")

        try:
            benchmark = MCPBenchmark()
            comparison_results = benchmark.run_comparison(
                finetuned_model_path="/app/checkpoints/mcp_finetuned/final_model"
            )

            # Save comparison results
            comparison_file = Path("/app/benchmarks/results/finetune_comparison.json")

            with open(comparison_file, 'w') as f:
                json.dump(comparison_results, f, indent=2)

            self.update_status("post_training_benchmarks", "completed", comparison_results)
            logger.info(f"Comparison results saved: {comparison_file}")

            return comparison_results

        except Exception as e:
            logger.error(f"Post-training benchmark failed: {e}")
            self.update_status("post_training_benchmarks", "failed", {"error": str(e)})
            raise

    def send_completion_email(self, report: dict):
        """Send email notification when training completes"""
        if not self.resend_api_key or not self.notification_email:
            logger.warning("Email notification skipped - credentials not configured")
            return

        try:
            success = report['success_criteria']['passed']
            status_emoji = "✅" if success else "⚠️"

            # Create HTML email
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
                <h1 style="color: #333;">{status_emoji} Phi-4 Fine-tuning Complete</h1>

                <div style="background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h2 style="margin-top: 0;">Training Summary</h2>
                    <ul>
                        <li><strong>Duration:</strong> {report['training_summary']['total_duration_hours']:.2f} hours</li>
                        <li><strong>Status:</strong> {'SUCCESS' if success else 'NEEDS REVIEW'}</li>
                        <li><strong>Started:</strong> {report['training_summary']['start_time']}</li>
                        <li><strong>Completed:</strong> {report['training_summary']['end_time']}</li>
                    </ul>
                </div>

                <div style="background: {'#d4edda' if success else '#fff3cd'}; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h2 style="margin-top: 0;">Performance Metrics</h2>
                    <ul>
                        <li><strong>Tool Accuracy:</strong> {report['validation_results'].get('accuracy', 0):.1f}% (Target: 95%)</li>
                        <li><strong>Parameter Accuracy:</strong> {report['validation_results'].get('param_accuracy', 0):.1f}%</li>
                        <li><strong>Quality Tier:</strong> {report['validation_results'].get('quality_tier', 'Unknown')}</li>
                    </ul>
                </div>

                <div style="background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h2 style="margin-top: 0;">Baseline vs Fine-tuned</h2>
                    <p>Improvement metrics available in detailed report.</p>
                </div>

                <div style="margin-top: 30px; padding: 15px; background: #f9f9f9; border-left: 4px solid #007bff;">
                    <p style="margin: 0;"><strong>Next Steps:</strong></p>
                    <ul style="margin-bottom: 0;">
                        <li>Review detailed results in Cloud Run logs</li>
                        <li>Download model from: <code>/app/checkpoints/mcp_finetuned/final_model/</code></li>
                        <li>{'Upload to Hugging Face: python3 cloudrun/upload_to_huggingface.py' if success else 'Review training parameters and re-run'}</li>
                        <li>{'Deploy to production' if success else 'Adjust hyperparameters'}</li>
                    </ul>
                </div>

                <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">

                <p style="color: #666; font-size: 12px;">
                    This is an automated notification from your Phi-4 fine-tuning pipeline on Google Cloud Run.
                </p>
            </body>
            </html>
            """

            params = {
                "from": "Agentic Flow <onboarding@resend.dev>",
                "to": [self.notification_email],
                "subject": f"{status_emoji} Phi-4 Fine-tuning Complete - {'Success' if success else 'Needs Review'}",
                "html": html_content,
                "tags": [
                    {"name": "project", "value": "phi4-finetuning"},
                    {"name": "status", "value": "success" if success else "review"}
                ]
            }

            email_response = resend.Emails.send(params)
            logger.info(f"Email notification sent successfully: {email_response}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def generate_final_report(self, baseline, validation, comparison):
        """Generate comprehensive final report"""
        logger.info("=" * 60)
        logger.info("PHASE 5: Final Report Generation")
        logger.info("=" * 60)

        report = {
            "training_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            },
            "baseline_performance": baseline,
            "validation_results": validation,
            "improvement_metrics": comparison,
            "success_criteria": {
                "tool_accuracy_target": 95.0,
                "tool_accuracy_achieved": validation.get("accuracy", 0),
                "passed": validation.get("accuracy", 0) >= 95.0
            }
        }

        # Save final report
        report_file = Path("/app/benchmarks/results/final_training_report.json")

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate markdown summary
        self._generate_markdown_report(report)

        # Send email notification
        self.send_completion_email(report)

        logger.info(f"Final report saved: {report_file}")
        return report

    def _generate_markdown_report(self, report):
        """Generate human-readable markdown report"""
        markdown = f"""# Phi-4 Fine-Tuning Results

## Training Summary
- **Start Time**: {report['training_summary']['start_time']}
- **End Time**: {report['training_summary']['end_time']}
- **Duration**: {report['training_summary']['total_duration_hours']:.2f} hours

## Baseline Performance
```json
{json.dumps(report['baseline_performance'], indent=2)}
```

## Validation Results
- **Tool Accuracy**: {report['validation_results'].get('accuracy', 0):.1f}%
- **Parameter Accuracy**: {report['validation_results'].get('param_accuracy', 0):.1f}%
- **Quality Tier**: {report['validation_results'].get('quality_tier', 'Unknown')}

## Success Criteria
- **Target**: {report['success_criteria']['tool_accuracy_target']}% tool accuracy
- **Achieved**: {report['success_criteria']['tool_accuracy_achieved']}%
- **Status**: {'✅ PASSED' if report['success_criteria']['passed'] else '❌ FAILED'}

## Improvement Metrics
```json
{json.dumps(report['improvement_metrics'], indent=2)}
```

---
Generated on Cloud Run GPU instance
"""

        markdown_file = Path("/app/benchmarks/results/TRAINING_REPORT.md")
        with open(markdown_file, 'w') as f:
            f.write(markdown)

        logger.info(f"Markdown report saved: {markdown_file}")

    def run_complete_pipeline(self):
        """Execute complete training and evaluation pipeline"""
        logger.info("=" * 80)
        logger.info("CLOUD RUN GPU TRAINING PIPELINE - PHI-4 FINE-TUNING")
        logger.info("=" * 80)

        try:
            # Phase 1: Baseline benchmarks
            baseline = self.run_baseline_benchmarks()

            # Phase 2: Fine-tuning
            training_metrics = self.run_finetuning()

            # Phase 3: Validation
            validation = self.run_validation()

            # Phase 4: Post-training benchmarks
            comparison = self.run_post_training_benchmarks()

            # Phase 5: Final report
            final_report = self.generate_final_report(baseline, validation, comparison)

            self.update_status("pipeline", "completed", final_report)

            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total duration: {final_report['training_summary']['total_duration_hours']:.2f} hours")
            logger.info(f"Tool accuracy: {validation.get('accuracy', 0):.1f}%")
            logger.info("=" * 80)

            return final_report

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.update_status("pipeline", "failed", {"error": str(e)})
            raise


def main():
    """Main entry point for Cloud Run"""
    logger.info("Starting Cloud Run training orchestrator...")

    orchestrator = CloudRunOrchestrator()

    try:
        final_report = orchestrator.run_complete_pipeline()

        # Keep container alive for result retrieval
        logger.info("Training complete. Container will remain available for result retrieval.")
        logger.info("Results available at: /app/benchmarks/results/")

        # Sleep to keep container alive (Cloud Run will handle shutdown)
        while True:
            time.sleep(60)
            logger.info("Container still running. Results available.")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
