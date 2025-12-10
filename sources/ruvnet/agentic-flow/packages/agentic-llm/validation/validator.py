#!/usr/bin/env python3
"""
Advanced Validation System with Overfitting Detection
For Phi-4 Agentic Training System
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    train_loss: float
    val_loss: float
    test_loss: float
    train_perplexity: float
    val_perplexity: float
    test_perplexity: float
    overfitting_ratio: float
    generalization_gap: float
    is_overfitting: bool


class OverfittingDetector:
    """Detect and prevent overfitting during training"""

    def __init__(
        self,
        overfitting_threshold: float = 0.15,
        window_size: int = 5
    ):
        self.overfitting_threshold = overfitting_threshold
        self.window_size = window_size
        self.train_losses = []
        self.val_losses = []

    def add_metrics(self, train_loss: float, val_loss: float):
        """Add training and validation losses"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def compute_overfitting_ratio(self) -> float:
        """Compute ratio of validation to training loss"""
        if len(self.val_losses) == 0 or len(self.train_losses) == 0:
            return 0.0

        recent_val = np.mean(self.val_losses[-self.window_size:])
        recent_train = np.mean(self.train_losses[-self.window_size:])

        if recent_train == 0:
            return 0.0

        ratio = (recent_val - recent_train) / recent_train
        return ratio

    def is_overfitting(self) -> bool:
        """Check if model is overfitting"""
        ratio = self.compute_overfitting_ratio()
        return ratio > self.overfitting_threshold

    def get_generalization_gap(self) -> float:
        """Compute generalization gap"""
        if len(self.val_losses) == 0 or len(self.train_losses) == 0:
            return 0.0

        return np.mean(self.val_losses) - np.mean(self.train_losses)

    def plot_learning_curves(self, save_path: str):
        """Plot training and validation learning curves"""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Training Loss", alpha=0.7)
        plt.plot(self.val_losses, label="Validation Loss", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        overfitting_ratios = [
            (self.val_losses[i] - self.train_losses[i]) / self.train_losses[i]
            if self.train_losses[i] != 0 else 0
            for i in range(len(self.train_losses))
        ]
        plt.plot(overfitting_ratios, label="Overfitting Ratio", color="red", alpha=0.7)
        plt.axhline(y=self.overfitting_threshold, color='orange', linestyle='--', label='Threshold')
        plt.xlabel("Epoch")
        plt.ylabel("Overfitting Ratio")
        plt.title("Overfitting Detection")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to {save_path}")


class MCPToolValidator:
    """Validate MCP tool calling accuracy"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def validate_tool_calls(self, test_cases: List[Dict]) -> Dict:
        """Validate model's ability to generate correct tool calls"""
        results = {
            "total": len(test_cases),
            "correct": 0,
            "partial": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "details": []
        }

        for i, test_case in enumerate(test_cases):
            prompt = test_case["input"]
            expected_tool = test_case.get("expected_tool", "")

            # Generate prediction
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1
                )

            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check if expected tool is in prediction
            if expected_tool.lower() in prediction.lower():
                results["correct"] += 1
                score = "correct"
            elif "<tool_use>" in prediction:
                results["partial"] += 1
                score = "partial"
            else:
                results["incorrect"] += 1
                score = "incorrect"

            results["details"].append({
                "input": prompt,
                "expected": expected_tool,
                "prediction": prediction,
                "score": score
            })

        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0
        return results


class AgenticWorkflowValidator:
    """Validate agentic workflow planning and execution"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def validate_workflow_planning(self, test_cases: List[Dict]) -> Dict:
        """Validate multi-step workflow planning"""
        results = {
            "total": len(test_cases),
            "planning_scores": [],
            "avg_planning_score": 0.0,
            "details": []
        }

        for test_case in test_cases:
            prompt = test_case["input"]
            expected_steps = test_case.get("expected_steps", [])

            # Generate prediction
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1
                )

            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Score based on step coverage
            steps_found = sum(1 for step in expected_steps if step.lower() in prediction.lower())
            planning_score = steps_found / len(expected_steps) if expected_steps else 0.0

            results["planning_scores"].append(planning_score)
            results["details"].append({
                "input": prompt,
                "expected_steps": expected_steps,
                "prediction": prediction,
                "score": planning_score
            })

        results["avg_planning_score"] = np.mean(results["planning_scores"]) if results["planning_scores"] else 0.0
        return results


class ComprehensiveValidator:
    """Main validation orchestrator"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "/app/validation/results"
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.overfitting_detector = OverfittingDetector()

    def run_full_validation(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict
    ) -> ValidationMetrics:
        """Run comprehensive validation suite"""
        logger.info("Running full validation suite...")

        # Extract metrics
        train_loss = train_metrics.get("loss", 0.0)
        val_loss = val_metrics.get("loss", 0.0)
        test_loss = test_metrics.get("loss", 0.0)

        # Compute perplexities
        train_perplexity = np.exp(train_loss)
        val_perplexity = np.exp(val_loss)
        test_perplexity = np.exp(test_loss)

        # Add to overfitting detector
        self.overfitting_detector.add_metrics(train_loss, val_loss)

        # Compute overfitting metrics
        overfitting_ratio = self.overfitting_detector.compute_overfitting_ratio()
        generalization_gap = self.overfitting_detector.get_generalization_gap()
        is_overfitting = self.overfitting_detector.is_overfitting()

        # Create metrics object
        metrics = ValidationMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            train_perplexity=train_perplexity,
            val_perplexity=val_perplexity,
            test_perplexity=test_perplexity,
            overfitting_ratio=overfitting_ratio,
            generalization_gap=generalization_gap,
            is_overfitting=is_overfitting
        )

        # Save metrics
        metrics_dict = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_perplexity": float(train_perplexity),
            "val_perplexity": float(val_perplexity),
            "test_perplexity": float(test_perplexity),
            "overfitting_ratio": float(overfitting_ratio),
            "generalization_gap": float(generalization_gap),
            "is_overfitting": is_overfitting
        }

        with open(self.output_dir / "validation_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)

        # Plot learning curves
        self.overfitting_detector.plot_learning_curves(
            str(self.output_dir / "learning_curves.png")
        )

        logger.info(f"Validation complete. Overfitting: {is_overfitting}")
        return metrics


def main():
    """Main validation entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Phi-4 model training")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--output-dir", type=str, default="/app/validation/results", help="Output directory")

    args = parser.parse_args()

    validator = ComprehensiveValidator(
        model_path=args.model_path,
        output_dir=args.output_dir
    )

    # Example metrics (would come from training)
    train_metrics = {"loss": 0.5}
    val_metrics = {"loss": 0.6}
    test_metrics = {"loss": 0.62}

    metrics = validator.run_full_validation(train_metrics, val_metrics, test_metrics)

    logger.info(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
