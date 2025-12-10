"""
Checkpoint saving and model persistence system.
Handles saving/loading model weights, training state, and configuration.
"""

import numpy as np
import json
import os
import pickle
import gzip
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import shutil


class Checkpoint:
    """
    Handles model checkpointing and persistence.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_frequency: int = 1,
        max_to_keep: int = 5
    ):
        """
        Initialize Checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            monitor: Metric to monitor for best model
            mode: "min" or "max" - whether lower or higher is better
            save_frequency: Save checkpoint every N epochs
            max_to_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_frequency = save_frequency
        self.max_to_keep = max_to_keep
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track best metric
        self.best = None
        if mode == "min":
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
        
        # Track saved checkpoints
        self.saved_checkpoints = []
    
    def save(
        self,
        epoch: int,
        model_params: Dict[str, np.ndarray],
        optimizer_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch
            model_params: Model parameters to save
            optimizer_state: Optimizer state (learning rates, momentum, etc.)
            metrics: Current metrics
            model_config: Model configuration
            training_config: Training configuration
            
        Returns:
            Path to saved checkpoint or None if not saved
        """
        # Check if should save
        if self.save_best_only and metrics:
            current = metrics.get(self.monitor)
            if current is None:
                print(f"Warning: Can't save checkpoint, metric '{self.monitor}' not found")
                return None
            
            if self.best is None:
                self.best = current
            elif not self.monitor_op(current, self.best):
                return None
            else:
                self.best = current
        
        if not self.save_best_only and epoch % self.save_frequency != 0:
            return None
        
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_params': model_params,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'best_metric': self.best
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        
        if model_config is not None:
            checkpoint['model_config'] = model_config
        
        if training_config is not None:
            checkpoint['training_config'] = training_config
        
        # Generate filename
        if self.save_best_only:
            filename = "best_model.pkl.gz"
        else:
            metric_str = f"{self.monitor}_{self.best:.4f}" if self.best is not None else ""
            filename = f"checkpoint_epoch_{epoch:04d}_{metric_str}.pkl.gz"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        self._save_checkpoint(checkpoint, filepath)
        
        # Manage checkpoint history
        if not self.save_best_only:
            self.saved_checkpoints.append(filepath)
            self._cleanup_old_checkpoints()
        
        print(f"Checkpoint saved: {filepath}")
        return filepath
    
    def load(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads best model.
            
        Returns:
            Dictionary containing checkpoint data
        """
        if checkpoint_path is None:
            # Try to load best model
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pkl.gz")
            if not os.path.exists(checkpoint_path):
                # Try to load latest checkpoint
                checkpoints = self.list_checkpoints()
                if checkpoints:
                    checkpoint_path = checkpoints[-1]
                else:
                    raise ValueError("No checkpoints found")
        
        return self._load_checkpoint(checkpoint_path)
    
    def load_latest(self) -> Dict[str, Any]:
        """Load the latest checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise ValueError("No checkpoints found")
        
        return self._load_checkpoint(checkpoints[-1])
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints sorted by creation time."""
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pkl.gz'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                checkpoints.append((filepath, os.path.getctime(filepath)))
        
        # Sort by creation time
        checkpoints.sort(key=lambda x: x[1])
        
        return [cp[0] for cp in checkpoints]
    
    def _save_checkpoint(self, checkpoint: Dict[str, Any], filepath: str):
        """Save checkpoint to compressed pickle file."""
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint from compressed pickle file."""
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only max_to_keep."""
        if len(self.saved_checkpoints) > self.max_to_keep:
            # Remove oldest checkpoints
            to_remove = self.saved_checkpoints[:-self.max_to_keep]
            for checkpoint in to_remove:
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)
                    print(f"Removed old checkpoint: {checkpoint}")
            
            self.saved_checkpoints = self.saved_checkpoints[-self.max_to_keep:]


class ModelSaver:
    """
    High-level model saving/loading with multiple formats support.
    """
    
    @staticmethod
    def save_model(
        model_params: Dict[str, np.ndarray],
        save_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "npz"
    ):
        """
        Save model in specified format.
        
        Args:
            model_params: Model parameters
            save_path: Path to save model
            model_config: Model configuration
            metadata: Additional metadata
            format: Save format ("npz", "json", "pkl")
        """
        # Prepare data
        save_data = {
            'model_params': model_params,
            'timestamp': datetime.now().isoformat(),
            'format_version': '1.0'
        }
        
        if model_config is not None:
            save_data['model_config'] = model_config
        
        if metadata is not None:
            save_data['metadata'] = metadata
        
        # Save based on format
        if format == "npz":
            # NumPy compressed format
            np_data = {}
            
            # Flatten parameters
            for name, param in model_params.items():
                np_data[f'param_{name}'] = param
            
            # Save config and metadata as JSON strings
            if model_config is not None:
                np_data['model_config'] = np.array(json.dumps(model_config))
            
            if metadata is not None:
                np_data['metadata'] = np.array(json.dumps(metadata))
            
            np_data['timestamp'] = np.array(save_data['timestamp'])
            
            np.savez_compressed(save_path, **np_data)
            
        elif format == "json":
            # JSON format (parameters as lists)
            json_data = {
                'model_params': {k: v.tolist() for k, v in model_params.items()},
                'model_config': model_config,
                'metadata': metadata,
                'timestamp': save_data['timestamp']
            }
            
            with open(save_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        elif format == "pkl":
            # Pickle format
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Model saved to: {save_path}")
    
    @staticmethod
    def load_model(load_path: str, format: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model from file.
        
        Args:
            load_path: Path to load model from
            format: Format to load (auto-detected if None)
            
        Returns:
            Dictionary containing model data
        """
        # Auto-detect format
        if format is None:
            if load_path.endswith('.npz'):
                format = 'npz'
            elif load_path.endswith('.json'):
                format = 'json'
            elif load_path.endswith('.pkl'):
                format = 'pkl'
            else:
                raise ValueError("Cannot auto-detect format, please specify")
        
        # Load based on format
        if format == "npz":
            data = np.load(load_path, allow_pickle=True)
            
            # Extract parameters
            model_params = {}
            for key in data.keys():
                if key.startswith('param_'):
                    param_name = key[6:]  # Remove 'param_' prefix
                    model_params[param_name] = data[key]
            
            # Extract config and metadata
            result = {'model_params': model_params}
            
            if 'model_config' in data:
                result['model_config'] = json.loads(str(data['model_config']))
            
            if 'metadata' in data:
                result['metadata'] = json.loads(str(data['metadata']))
            
            if 'timestamp' in data:
                result['timestamp'] = str(data['timestamp'])
            
            return result
            
        elif format == "json":
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            if 'model_params' in data:
                data['model_params'] = {
                    k: np.array(v) for k, v in data['model_params'].items()
                }
            
            return data
            
        elif format == "pkl":
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unknown format: {format}")


class TrainingState:
    """
    Manages complete training state for resuming interrupted training.
    """
    
    def __init__(self, state_file: str = "training_state.json"):
        """
        Initialize TrainingState.
        
        Args:
            state_file: Path to state file
        """
        self.state_file = state_file
        self.state = {
            'current_epoch': 0,
            'total_epochs': 0,
            'best_metric': None,
            'best_epoch': 0,
            'training_history': {},
            'random_state': None
        }
    
    def update(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
        best_metric: Optional[float] = None,
        best_epoch: Optional[int] = None
    ):
        """Update training state."""
        self.state['current_epoch'] = epoch
        self.state['total_epochs'] = total_epochs
        
        if best_metric is not None:
            self.state['best_metric'] = best_metric
        
        if best_epoch is not None:
            self.state['best_epoch'] = best_epoch
        
        # Update history
        for name, value in metrics.items():
            if name not in self.state['training_history']:
                self.state['training_history'][name] = []
            self.state['training_history'][name].append(value)
        
        # Save random state
        self.state['random_state'] = {
            'numpy_state': pickle.dumps(np.random.get_state()),
        }
        
        self.save()
    
    def save(self):
        """Save state to file."""
        # Convert non-serializable data
        save_state = self.state.copy()
        if 'random_state' in save_state and save_state['random_state'] is not None:
            save_state['random_state'] = {
                k: v.hex() if isinstance(v, bytes) else v
                for k, v in save_state['random_state'].items()
            }
        
        with open(self.state_file, 'w') as f:
            json.dump(save_state, f, indent=2)
    
    def load(self) -> bool:
        """
        Load state from file.
        
        Returns:
            True if state was loaded successfully
        """
        if not os.path.exists(self.state_file):
            return False
        
        try:
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            
            # Restore random state
            if 'random_state' in self.state and self.state['random_state'] is not None:
                if 'numpy_state' in self.state['random_state']:
                    numpy_state = bytes.fromhex(self.state['random_state']['numpy_state'])
                    np.random.set_state(pickle.loads(numpy_state))
            
            return True
            
        except Exception as e:
            print(f"Error loading training state: {e}")
            return False
    
    def get_resume_epoch(self) -> int:
        """Get epoch to resume from."""
        return self.state['current_epoch'] + 1
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.state['training_history']