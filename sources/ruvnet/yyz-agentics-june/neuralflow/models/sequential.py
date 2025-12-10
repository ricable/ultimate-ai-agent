"""
Sequential model implementation
"""
import numpy as np
from typing import List, Union, Optional, Tuple
from ..core.tensor import Tensor
from ..core.losses import get_loss, Loss
from ..core.optimizers import get_optimizer, Optimizer


class Sequential:
    """
    Sequential model - linear stack of layers.
    
    Example:
        model = Sequential([
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
    """
    
    def __init__(self, layers: Optional[List] = None):
        self.layers = layers or []
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        self.history = {'loss': [], 'val_loss': []}
        self.compiled = False
    
    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)
    
    def compile(self, optimizer: Union[str, Optimizer], 
                loss: Union[str, Loss],
                metrics: Optional[List[str]] = None):
        """
        Compile the model for training.
        
        Parameters:
            optimizer: Optimizer to use
            loss: Loss function to use
            metrics: List of metrics to track
        """
        self.optimizer = get_optimizer(optimizer)
        self.loss_fn = get_loss(loss)
        self.metrics = metrics or []
        
        # Collect all parameters
        parameters = []
        for layer in self.layers:
            if hasattr(layer, 'get_parameters'):
                parameters.extend(layer.get_parameters())
        
        self.optimizer.add_parameters(parameters)
        self.compiled = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def train_step(self, x: Tensor, y: Tensor) -> Tuple[float, dict]:
        """Single training step."""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.forward(x)
        
        # Calculate loss
        loss = self.loss_fn(y, predictions)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        # Calculate metrics
        metrics = {'loss': float(loss.data)}
        
        if 'accuracy' in self.metrics:
            if len(y.shape) > 1:  # One-hot encoded
                accuracy = np.mean(np.argmax(predictions.data, axis=-1) == np.argmax(y.data, axis=-1))
            else:  # Binary or sparse categorical
                if predictions.shape[-1] == 1:  # Binary
                    accuracy = np.mean((predictions.data > 0.5) == y.data)
                else:  # Sparse categorical
                    accuracy = np.mean(np.argmax(predictions.data, axis=-1) == y.data)
            metrics['accuracy'] = float(accuracy)
        
        return float(loss.data), metrics
    
    def evaluate(self, x: Tensor, y: Tensor, batch_size: int = 32) -> dict:
        """Evaluate the model on given data."""
        # Set layers to evaluation mode
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
        
        total_loss = 0
        total_samples = 0
        metrics_sum = {}
        
        # Process in batches
        n_samples = x.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            x_batch = Tensor(x.data[start_idx:end_idx])
            y_batch = Tensor(y.data[start_idx:end_idx])
            
            # Forward pass
            predictions = self.forward(x_batch)
            
            # Calculate loss
            loss = self.loss_fn(y_batch, predictions)
            
            batch_size_actual = end_idx - start_idx
            total_loss += float(loss.data) * batch_size_actual
            total_samples += batch_size_actual
            
            # Calculate metrics
            if 'accuracy' in self.metrics:
                if len(y.shape) > 1:  # One-hot encoded
                    accuracy = np.sum(np.argmax(predictions.data, axis=-1) == np.argmax(y_batch.data, axis=-1))
                else:  # Binary or sparse categorical
                    if predictions.shape[-1] == 1:  # Binary
                        accuracy = np.sum((predictions.data > 0.5) == y_batch.data)
                    else:  # Sparse categorical
                        accuracy = np.sum(np.argmax(predictions.data, axis=-1) == y_batch.data)
                
                if 'accuracy' not in metrics_sum:
                    metrics_sum['accuracy'] = 0
                metrics_sum['accuracy'] += accuracy
        
        # Set layers back to training mode
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
        
        # Calculate averages
        metrics = {'loss': total_loss / total_samples}
        for metric_name, metric_sum in metrics_sum.items():
            metrics[metric_name] = metric_sum / total_samples
        
        return metrics
    
    def fit(self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray],
            epochs: int = 10,
            batch_size: int = 32,
            validation_data: Optional[Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]] = None,
            verbose: int = 1,
            shuffle: bool = True) -> dict:
        """
        Train the model.
        
        Parameters:
            x: Training data
            y: Training labels
            epochs: Number of epochs to train
            batch_size: Batch size
            validation_data: Validation data (x_val, y_val)
            verbose: Verbosity mode
            shuffle: Whether to shuffle data between epochs
        
        Returns:
            Training history
        """
        if not self.compiled:
            raise ValueError("Model must be compiled before training")
        
        # Convert to tensors if necessary
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if isinstance(y, np.ndarray):
            y = Tensor(y)
        
        n_samples = x.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                x.data = x.data[indices]
                y.data = y.data[indices]
            
            epoch_loss = 0
            epoch_metrics = {}
            
            # Train on batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                x_batch = Tensor(x.data[start_idx:end_idx], requires_grad=True)
                y_batch = Tensor(y.data[start_idx:end_idx])
                
                loss, metrics = self.train_step(x_batch, y_batch)
                epoch_loss += loss
                
                for metric_name, metric_value in metrics.items():
                    if metric_name not in epoch_metrics:
                        epoch_metrics[metric_name] = 0
                    epoch_metrics[metric_name] += metric_value
            
            # Calculate epoch averages
            epoch_loss /= n_batches
            for metric_name in epoch_metrics:
                epoch_metrics[metric_name] /= n_batches
            
            self.history['loss'].append(epoch_loss)
            
            # Validation
            if validation_data is not None:
                x_val, y_val = validation_data
                if isinstance(x_val, np.ndarray):
                    x_val = Tensor(x_val)
                if isinstance(y_val, np.ndarray):
                    y_val = Tensor(y_val)
                
                val_metrics = self.evaluate(x_val, y_val, batch_size)
                self.history['val_loss'].append(val_metrics['loss'])
                
                for metric_name, metric_value in val_metrics.items():
                    key = f'val_{metric_name}'
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(metric_value)
            
            # Print progress
            if verbose > 0:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}"
                for metric_name, metric_value in epoch_metrics.items():
                    if metric_name != 'loss':
                        msg += f" - {metric_name}: {metric_value:.4f}"
                
                if validation_data is not None:
                    msg += f" - val_loss: {val_metrics['loss']:.4f}"
                    for metric_name, metric_value in val_metrics.items():
                        if metric_name != 'loss':
                            msg += f" - val_{metric_name}: {metric_value:.4f}"
                
                print(msg)
        
        return self.history
    
    def predict(self, x: Union[Tensor, np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Parameters:
            x: Input data
            batch_size: Batch size for prediction
            
        Returns:
            Predictions as numpy array
        """
        # Set layers to evaluation mode
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
        
        # Convert to tensor if necessary
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        
        n_samples = x.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        predictions = []
        
        # Process in batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            x_batch = Tensor(x.data[start_idx:end_idx])
            
            # Forward pass
            batch_predictions = self.forward(x_batch)
            predictions.append(batch_predictions.data)
        
        # Set layers back to training mode
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
        
        return np.concatenate(predictions, axis=0)
    
    def summary(self):
        """Print model summary."""
        print("Model: Sequential")
        print("=" * 65)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
        print("=" * 65)
        
        total_params = 0
        trainable_params = 0
        
        # Dummy input to trace shapes
        input_shape = (1, 784)  # Default shape, will be overridden
        x = Tensor.zeros(input_shape)
        
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.__class__.__name__}_{i}"
            
            # Get output shape
            x = layer(x)
            output_shape = str(x.shape)
            
            # Count parameters
            param_count = 0
            if hasattr(layer, 'get_parameters'):
                params = layer.get_parameters()
                for param in params:
                    param_count += param.data.size
                    if param.requires_grad:
                        trainable_params += param.data.size
            
            total_params += param_count
            
            print(f"{layer_name:<25} {output_shape:<20} {param_count:<15,}")
        
        print("=" * 65)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 65)