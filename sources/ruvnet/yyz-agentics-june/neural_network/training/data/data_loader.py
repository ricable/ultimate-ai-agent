"""
Data loading and batching system for neural network training.
Supports multiple datasets, data augmentation, and efficient CPU-based batch processing.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import json
import os
from collections import defaultdict


class Dataset(ABC):
    """Abstract base class for datasets."""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input, target)
        """
        pass
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (batch_inputs, batch_targets)
        """
        inputs = []
        targets = []
        
        for idx in indices:
            input_data, target_data = self[idx]
            inputs.append(input_data)
            targets.append(target_data)
        
        return np.array(inputs), np.array(targets)


class DataLoader:
    """
    Efficient data loader with batching, shuffling, and prefetching.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        augmentation_fn: Optional[Callable] = None
    ):
        """
        Initialize DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop the last incomplete batch
            num_workers: Number of parallel workers (currently not used, for future multiprocessing)
            collate_fn: Function to collate samples into batches
            augmentation_fn: Function to apply data augmentation
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.collate_fn = collate_fn or self._default_collate
        self.augmentation_fn = augmentation_fn
        
        self._indices = None
        self._reset_indices()
        
    def _reset_indices(self):
        """Reset and optionally shuffle indices."""
        self._indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self._indices)
            
    def _default_collate(self, batch: List[Tuple[np.ndarray, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Default collate function."""
        inputs = np.array([item[0] for item in batch])
        targets = np.array([item[1] for item in batch])
        return inputs, targets
        
    def __len__(self) -> int:
        """Return the number of batches."""
        n_samples = len(self.dataset)
        n_batches = n_samples // self.batch_size
        if not self.drop_last and n_samples % self.batch_size != 0:
            n_batches += 1
        return n_batches
    
    def __iter__(self):
        """Iterate over batches."""
        self._reset_indices()
        
        for i in range(0, len(self._indices), self.batch_size):
            batch_indices = self._indices[i:i + self.batch_size]
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
                
            # Get batch samples
            batch = []
            for idx in batch_indices:
                sample = self.dataset[idx]
                
                # Apply augmentation if provided
                if self.augmentation_fn is not None:
                    sample = self.augmentation_fn(sample)
                    
                batch.append(sample)
            
            # Collate batch
            yield self.collate_fn(batch)


class ArrayDataset(Dataset):
    """Simple dataset for numpy arrays."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transforms: Optional[List[Callable]] = None):
        """
        Initialize ArrayDataset.
        
        Args:
            X: Input array of shape (n_samples, ...)
            y: Target array of shape (n_samples, ...)
            transforms: List of transforms to apply to inputs
        """
        assert len(X) == len(y), "Inputs and targets must have the same length"
        self.X = X
        self.y = y
        self.transforms = transforms or []
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self.X[idx].copy()
        y = self.y[idx]
        
        # Apply transforms
        for transform in self.transforms:
            x = transform(x)
            
        return x, y


class MultiDatasetLoader:
    """
    Data loader that can handle multiple datasets with different sampling strategies.
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        batch_size: int = 32,
        sampling_strategy: str = "uniform",
        dataset_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize MultiDatasetLoader.
        
        Args:
            datasets: Dictionary of dataset_name -> Dataset
            batch_size: Number of samples per batch
            sampling_strategy: How to sample from datasets ("uniform", "weighted", "sequential")
            dataset_weights: Weights for weighted sampling
            **kwargs: Additional arguments passed to individual DataLoaders
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.sampling_strategy = sampling_strategy
        self.dataset_weights = dataset_weights or {name: 1.0 for name in datasets}
        
        # Create individual data loaders
        self.loaders = {
            name: DataLoader(dataset, batch_size=batch_size, **kwargs)
            for name, dataset in datasets.items()
        }
        
        # Normalize weights
        total_weight = sum(self.dataset_weights.values())
        self.dataset_weights = {
            name: w / total_weight 
            for name, w in self.dataset_weights.items()
        }
        
    def __len__(self) -> int:
        """Return total number of batches across all datasets."""
        return sum(len(loader) for loader in self.loaders.values())
    
    def __iter__(self):
        """Iterate over batches from multiple datasets."""
        if self.sampling_strategy == "sequential":
            # Sample from each dataset sequentially
            for name, loader in self.loaders.items():
                for batch in loader:
                    yield name, batch
                    
        elif self.sampling_strategy == "uniform":
            # Sample uniformly from all datasets
            iterators = {name: iter(loader) for name, loader in self.loaders.items()}
            active_datasets = list(iterators.keys())
            
            while active_datasets:
                # Randomly select a dataset
                dataset_name = np.random.choice(active_datasets)
                
                try:
                    batch = next(iterators[dataset_name])
                    yield dataset_name, batch
                except StopIteration:
                    active_datasets.remove(dataset_name)
                    
        elif self.sampling_strategy == "weighted":
            # Sample based on weights
            iterators = {name: iter(loader) for name, loader in self.loaders.items()}
            active_datasets = list(iterators.keys())
            
            while active_datasets:
                # Select dataset based on weights
                weights = [self.dataset_weights[name] for name in active_datasets]
                weights = np.array(weights) / np.sum(weights)  # Renormalize
                dataset_name = np.random.choice(active_datasets, p=weights)
                
                try:
                    batch = next(iterators[dataset_name])
                    yield dataset_name, batch
                except StopIteration:
                    active_datasets.remove(dataset_name)


# Data augmentation functions
class DataAugmentation:
    """Collection of data augmentation techniques."""
    
    @staticmethod
    def add_noise(x: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to input."""
        noise = np.random.normal(0, noise_std, x.shape)
        return x + noise
    
    @staticmethod
    def random_scale(x: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Randomly scale input."""
        scale = np.random.uniform(*scale_range)
        return x * scale
    
    @staticmethod
    def random_flip(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Randomly flip input along an axis."""
        if np.random.random() > 0.5:
            return np.flip(x, axis=axis)
        return x
    
    @staticmethod
    def dropout(x: np.ndarray, drop_rate: float = 0.1) -> np.ndarray:
        """Apply dropout to input."""
        mask = np.random.random(x.shape) > drop_rate
        return x * mask / (1 - drop_rate)
    
    @staticmethod
    def compose(*augmentations):
        """Compose multiple augmentation functions."""
        def composed(x):
            for aug in augmentations:
                x = aug(x)
            return x
        return composed


# Utility functions for common datasets
def create_synthetic_dataset(
    n_samples: int = 1000,
    input_dim: int = 10,
    output_dim: int = 1,
    noise_level: float = 0.1,
    dataset_type: str = "regression"
) -> ArrayDataset:
    """
    Create a synthetic dataset for testing.
    
    Args:
        n_samples: Number of samples
        input_dim: Input dimension
        output_dim: Output dimension
        noise_level: Amount of noise to add
        dataset_type: Type of dataset ("regression" or "classification")
        
    Returns:
        ArrayDataset instance
    """
    # Generate random inputs
    X = np.random.randn(n_samples, input_dim)
    
    if dataset_type == "regression":
        # Generate linear relationship with noise
        W = np.random.randn(input_dim, output_dim)
        y = X @ W + np.random.randn(n_samples, output_dim) * noise_level
        
    elif dataset_type == "classification":
        # Generate classification data
        W = np.random.randn(input_dim, output_dim)
        logits = X @ W
        if output_dim == 1:
            # Binary classification
            y = (logits > 0).astype(np.float32)
        else:
            # Multi-class classification
            y = np.argmax(logits, axis=1)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    return ArrayDataset(X, y)


def train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    shuffle: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and test sets.
    
    Args:
        dataset: Dataset to split
        test_size: Fraction of data to use for testing
        shuffle: Whether to shuffle before splitting
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Create subset datasets
    if isinstance(dataset, ArrayDataset):
        train_dataset = ArrayDataset(
            dataset.X[train_indices],
            dataset.y[train_indices],
            dataset.transforms
        )
        test_dataset = ArrayDataset(
            dataset.X[test_indices],
            dataset.y[test_indices],
            dataset.transforms
        )
    else:
        # For custom datasets, create a wrapper
        class SubsetDataset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        train_dataset = SubsetDataset(dataset, train_indices)
        test_dataset = SubsetDataset(dataset, test_indices)
    
    return train_dataset, test_dataset