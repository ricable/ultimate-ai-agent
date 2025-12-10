"""
Visualization utilities for NeuralFlow
"""
import numpy as np
from typing import Optional, List, Tuple


def plot_history(history: dict, metrics: Optional[List[str]] = None) -> str:
    """
    Create ASCII plot of training history.
    
    Parameters:
        history: Training history dictionary
        metrics: Metrics to plot
    
    Returns:
        ASCII plot as string
    """
    if metrics is None:
        metrics = ['loss']
    
    plots = []
    
    for metric in metrics:
        if metric not in history:
            continue
        
        values = history[metric]
        val_metric = f'val_{metric}'
        val_values = history.get(val_metric, None)
        
        # Create ASCII plot
        plot = f"\n{metric.upper()}\n"
        plot += "=" * 50 + "\n"
        
        # Find min and max for scaling
        all_values = values.copy()
        if val_values:
            all_values.extend(val_values)
        
        if not all_values:
            continue
            
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val
        
        if range_val == 0:
            range_val = 1
        
        # Plot height
        height = 10
        width = min(len(values), 50)
        
        # Create plot grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot training values
        for i in range(min(len(values), width)):
            val = values[i * len(values) // width]
            y = int((1 - (val - min_val) / range_val) * (height - 1))
            y = max(0, min(height - 1, y))
            grid[y][i] = '*'
        
        # Plot validation values if available
        if val_values:
            for i in range(min(len(val_values), width)):
                val = val_values[i * len(val_values) // width]
                y = int((1 - (val - min_val) / range_val) * (height - 1))
                y = max(0, min(height - 1, y))
                if grid[y][i] == ' ':
                    grid[y][i] = 'o'
                else:
                    grid[y][i] = 'x'
        
        # Add axis labels
        plot += f"{max_val:.4f} |" + "\n"
        for row in grid:
            plot += "       |" + ''.join(row) + "\n"
        plot += f"{min_val:.4f} |" + '_' * width + "\n"
        plot += "       0" + " " * (width - 10) + f"Epoch {len(values)}\n"
        
        if val_values:
            plot += "\n  * = training, o = validation, x = both\n"
        
        plots.append(plot)
    
    return '\n'.join(plots)


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                      suffix: str = '', decimals: int = 1, 
                      length: int = 50, fill: str = '█') -> None:
    """
    Print progress bar.
    
    Parameters:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Number of decimals in percent
        length: Character length of bar
        fill: Bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    
    # Print new line on complete
    if iteration == total:
        print()


def visualize_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                         n_samples: int = 10) -> str:
    """
    Create ASCII visualization of predictions vs true labels.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        n_samples: Number of samples to show
    
    Returns:
        ASCII visualization
    """
    viz = "\nPREDICTIONS VISUALIZATION\n"
    viz += "=" * 40 + "\n"
    viz += "Sample | True | Pred | Correct\n"
    viz += "-" * 40 + "\n"
    
    n_samples = min(n_samples, len(y_true))
    
    for i in range(n_samples):
        true_label = int(y_true[i]) if y_true[i].shape == () else int(np.argmax(y_true[i]))
        pred_label = int(y_pred[i]) if y_pred[i].shape == () else int(np.argmax(y_pred[i]))
        correct = "✓" if true_label == pred_label else "✗"
        
        viz += f"{i:6d} | {true_label:4d} | {pred_label:4d} | {correct:^7s}\n"
    
    # Calculate accuracy
    if len(y_true.shape) > 1:  # One-hot encoded
        accuracy = np.mean(np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1))
    else:
        accuracy = np.mean(y_true == y_pred)
    
    viz += "-" * 40 + "\n"
    viz += f"Overall Accuracy: {accuracy:.2%}\n"
    
    return viz


def visualize_layer_weights(weights: np.ndarray, max_width: int = 50) -> str:
    """
    Create ASCII visualization of layer weights.
    
    Parameters:
        weights: Weight matrix
        max_width: Maximum width of visualization
    
    Returns:
        ASCII visualization
    """
    viz = "\nLAYER WEIGHTS VISUALIZATION\n"
    viz += "=" * max_width + "\n"
    viz += f"Shape: {weights.shape}\n"
    viz += f"Min: {weights.min():.4f}, Max: {weights.max():.4f}\n"
    viz += f"Mean: {weights.mean():.4f}, Std: {weights.std():.4f}\n"
    viz += "-" * max_width + "\n"
    
    # Create heatmap
    if len(weights.shape) == 2:
        h, w = weights.shape
        
        # Subsample if too large
        if w > max_width:
            step = w // max_width
            weights = weights[:, ::step]
            w = weights.shape[1]
        
        if h > 20:
            step = h // 20
            weights = weights[::step, :]
            h = weights.shape[0]
        
        # Normalize to 0-9
        normalized = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        normalized = (normalized * 9).astype(int)
        
        # Create visualization
        for i in range(h):
            row = ''.join(str(normalized[i, j]) for j in range(w))
            viz += row + "\n"
    else:
        viz += f"Cannot visualize weights with shape {weights.shape}\n"
    
    return viz


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                    n_classes: Optional[int] = None) -> str:
    """
    Create ASCII confusion matrix.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes
    
    Returns:
        ASCII confusion matrix
    """
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=-1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    
    # Calculate confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    
    # Create visualization
    viz = "\nCONFUSION MATRIX\n"
    viz += "=" * (n_classes * 6 + 10) + "\n"
    
    # Header
    viz += "True\\Pred |"
    for i in range(n_classes):
        viz += f"{i:^5d}|"
    viz += "\n"
    viz += "-" * (n_classes * 6 + 10) + "\n"
    
    # Matrix
    for i in range(n_classes):
        viz += f"{i:^9d}|"
        for j in range(n_classes):
            viz += f"{cm[i, j]:^5d}|"
        viz += "\n"
    
    # Summary statistics
    viz += "=" * (n_classes * 6 + 10) + "\n"
    accuracy = np.trace(cm) / cm.sum()
    viz += f"Accuracy: {accuracy:.2%}\n"
    
    return viz