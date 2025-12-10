"""
Training metrics and progress monitoring system.
Provides various metrics for tracking model performance during training.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from collections import defaultdict
import time
import json


class Metric:
    """Base class for all metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self._sum = 0.0
        self._count = 0
        self._values = []
    
    def update(self, value: float, n: int = 1):
        """
        Update metric with new value.
        
        Args:
            value: Metric value
            n: Number of samples (for averaging)
        """
        self._sum += value * n
        self._count += n
        self._values.append(value)
    
    def compute(self) -> float:
        """Compute the metric value."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count
    
    def get_history(self) -> List[float]:
        """Get history of metric values."""
        return self._values
    
    def __repr__(self):
        return f"{self.name}: {self.compute():.4f}"


class Loss(Metric):
    """Loss metric."""
    
    def __init__(self):
        super().__init__("loss")


class Accuracy(Metric):
    """Accuracy metric for classification."""
    
    def __init__(self):
        super().__init__("accuracy")
        self._correct = 0
        self._total = 0
    
    def reset(self):
        super().reset()
        self._correct = 0
        self._total = 0
    
    def update_from_predictions(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Update accuracy from predictions and targets.
        
        Args:
            y_pred: Predicted labels or probabilities
            y_true: True labels
        """
        # Handle different prediction formats
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Multi-class predictions (logits or probabilities)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            # Binary classification
            y_pred = (y_pred > 0.5).astype(int).squeeze()
        
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            # One-hot encoded targets
            y_true = np.argmax(y_true, axis=1)
        else:
            y_true = y_true.squeeze()
        
        correct = np.sum(y_pred == y_true)
        total = len(y_true)
        
        self._correct += correct
        self._total += total
        
        accuracy = correct / total if total > 0 else 0
        self.update(accuracy, total)
    
    def compute(self) -> float:
        """Compute overall accuracy."""
        if self._total == 0:
            return 0.0
        return self._correct / self._total


class MeanSquaredError(Metric):
    """Mean Squared Error metric for regression."""
    
    def __init__(self):
        super().__init__("mse")
    
    def update_from_predictions(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Update MSE from predictions and targets."""
        mse = np.mean((y_pred - y_true) ** 2)
        self.update(mse, len(y_true))


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric for regression."""
    
    def __init__(self):
        super().__init__("mae")
    
    def update_from_predictions(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Update MAE from predictions and targets."""
        mae = np.mean(np.abs(y_pred - y_true))
        self.update(mae, len(y_true))


class F1Score(Metric):
    """F1 Score metric for binary classification."""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__("f1_score")
        self.threshold = threshold
        self._tp = 0  # True positives
        self._fp = 0  # False positives
        self._fn = 0  # False negatives
    
    def reset(self):
        super().reset()
        self._tp = 0
        self._fp = 0
        self._fn = 0
    
    def update_from_predictions(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Update F1 score from predictions and targets."""
        # Convert to binary predictions
        y_pred_binary = (y_pred > self.threshold).astype(int).squeeze()
        y_true = y_true.astype(int).squeeze()
        
        self._tp += np.sum((y_pred_binary == 1) & (y_true == 1))
        self._fp += np.sum((y_pred_binary == 1) & (y_true == 0))
        self._fn += np.sum((y_pred_binary == 0) & (y_true == 1))
        
        # Calculate F1 for this batch
        precision = self._tp / (self._tp + self._fp) if (self._tp + self._fp) > 0 else 0
        recall = self._tp / (self._tp + self._fn) if (self._tp + self._fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.update(f1, len(y_true))
    
    def compute(self) -> float:
        """Compute overall F1 score."""
        precision = self._tp / (self._tp + self._fp) if (self._tp + self._fp) > 0 else 0
        recall = self._tp / (self._tp + self._fn) if (self._tp + self._fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1


class MetricTracker:
    """
    Tracks multiple metrics during training and provides monitoring capabilities.
    """
    
    def __init__(self, metrics: List[Metric]):
        """
        Initialize MetricTracker.
        
        Args:
            metrics: List of metrics to track
        """
        self.metrics = {metric.name: metric for metric in metrics}
        self.history = defaultdict(list)
        self.epoch_start_time = None
        self.batch_times = []
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
        self.batch_times = []
    
    def update(self, metric_name: str, value: float, n: int = 1):
        """Update a specific metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name].update(value, n)
    
    def update_from_predictions(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Update metrics that compute from predictions."""
        for metric in self.metrics.values():
            if hasattr(metric, 'update_from_predictions'):
                metric.update_from_predictions(y_pred, y_true)
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self.reset()
    
    def end_epoch(self):
        """Mark the end of an epoch and save metrics."""
        epoch_time = time.time() - self.epoch_start_time
        
        # Save metric values
        for name, metric in self.metrics.items():
            self.history[name].append(metric.compute())
        
        self.history['epoch_time'].append(epoch_time)
        
        return self.get_epoch_metrics()
    
    def start_batch(self):
        """Mark the start of a batch."""
        self._batch_start_time = time.time()
    
    def end_batch(self):
        """Mark the end of a batch."""
        batch_time = time.time() - self._batch_start_time
        self.batch_times.append(batch_time)
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get current epoch metrics."""
        metrics = {}
        for name, metric in self.metrics.items():
            metrics[name] = metric.compute()
        
        # Add timing metrics
        if self.batch_times:
            metrics['avg_batch_time'] = np.mean(self.batch_times)
            metrics['total_batch_time'] = np.sum(self.batch_times)
        
        return metrics
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full history of all metrics."""
        return dict(self.history)
    
    def print_epoch_summary(self, epoch: int, total_epochs: int, prefix: str = ""):
        """Print a summary of the current epoch."""
        metrics = self.get_epoch_metrics()
        
        # Format metrics string
        metric_strs = []
        for name, value in metrics.items():
            if name not in ['avg_batch_time', 'total_batch_time']:
                metric_strs.append(f"{name}: {value:.4f}")
        
        metrics_str = " - ".join(metric_strs)
        
        # Print summary
        print(f"{prefix}Epoch [{epoch}/{total_epochs}] - {metrics_str}")
        
        # Print timing info
        if 'avg_batch_time' in metrics:
            print(f"{prefix}  Avg batch time: {metrics['avg_batch_time']:.3f}s")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 1e-4,
        restore_best_weights: bool = True
    ):
        """
        Initialize EarlyStopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            mode: "min" or "max" - whether lower or higher is better
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.best_epoch = 0
        
        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model_params: Optional[Dict] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch
            logs: Dictionary of metrics
            model_params: Current model parameters (for restoring best weights)
            
        Returns:
            True if training should stop
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: Early stopping conditioned on metric '{self.monitor}' "
                  f"which is not available. Available metrics are: {list(logs.keys())}")
            return False
        
        if self.best is None:
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights and model_params is not None:
                self.best_weights = {k: v.copy() for k, v in model_params.items()}
        
        elif self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights and model_params is not None:
                self.best_weights = {k: v.copy() for k, v in model_params.items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        
        return False
    
    def get_best_weights(self) -> Optional[Dict]:
        """Get the best model weights if available."""
        return self.best_weights
    
    def on_train_end(self):
        """Called when training ends."""
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch}")
            print(f"Best {self.monitor}: {self.best:.4f} at epoch {self.best_epoch}")


class ProgressBar:
    """Simple progress bar for training visualization."""
    
    def __init__(self, total: int, width: int = 50):
        """
        Initialize ProgressBar.
        
        Args:
            total: Total number of steps
            width: Width of the progress bar
        """
        self.total = total
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1, postfix: Optional[Dict[str, float]] = None):
        """Update progress bar."""
        self.current = min(self.current + n, self.total)
        
        # Calculate progress
        progress = self.current / self.total
        filled = int(self.width * progress)
        
        # Create bar
        bar = '=' * filled + '>' + '-' * (self.width - filled - 1)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        eta = elapsed / progress - elapsed if progress > 0 else 0
        
        # Format postfix
        postfix_str = ""
        if postfix:
            postfix_items = [f"{k}: {v:.4f}" for k, v in postfix.items()]
            postfix_str = " - " + " - ".join(postfix_items)
        
        # Print progress
        print(f'\r[{bar}] {self.current}/{self.total} - '
              f'{elapsed:.1f}s - ETA: {eta:.1f}s{postfix_str}', end='', flush=True)
    
    def close(self):
        """Close progress bar."""
        print()  # New line


def create_metric_plots(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Create simple text-based plots of metrics (since we're CPU-only).
    
    Args:
        history: Dictionary of metric histories
        save_path: Optional path to save the plots
    """
    output = []
    
    for metric_name, values in history.items():
        if not values or metric_name == 'epoch_time':
            continue
            
        output.append(f"\n{metric_name.upper()} History:")
        output.append("=" * 50)
        
        # Find min and max for scaling
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        
        if range_val == 0:
            range_val = 1
        
        # Create simple ASCII plot
        height = 10
        width = min(len(values), 50)
        
        # Sample values if too many
        if len(values) > width:
            indices = np.linspace(0, len(values) - 1, width, dtype=int)
            sampled_values = [values[i] for i in indices]
        else:
            sampled_values = values
        
        # Create plot
        for h in range(height, -1, -1):
            line = ""
            threshold = min_val + (h / height) * range_val
            
            for val in sampled_values:
                if val >= threshold:
                    line += "*"
                else:
                    line += " "
            
            output.append(f"{threshold:>8.4f} |{line}")
        
        output.append(f"{'':>8} " + "-" * (width + 1))
        output.append(f"{'Epochs':>8} 0" + " " * (width - 10) + f"{len(values)}")
        
        # Summary statistics
        output.append(f"\nMin: {min_val:.4f}, Max: {max_val:.4f}, Final: {values[-1]:.4f}")
    
    result = "\n".join(output)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(result)
    
    return result