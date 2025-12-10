"""
Functional API Model implementation
"""
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Callable
from ..core.tensor import Tensor
from ..core.losses import get_loss, Loss
from ..core.optimizers import get_optimizer, Optimizer


class Model:
    """
    Model class for functional API.
    
    Allows building complex models with multiple inputs/outputs,
    shared layers, and non-sequential connections.
    
    Example:
        # Input layers
        input1 = Input(shape=(784,))
        input2 = Input(shape=(10,))
        
        # Shared layer
        shared = Dense(64, activation='relu')
        
        # Process inputs
        x1 = shared(input1)
        x2 = shared(input2)
        
        # Merge
        merged = Concatenate()([x1, x2])
        
        # Output
        output = Dense(10, activation='softmax')(merged)
        
        # Create model
        model = Model(inputs=[input1, input2], outputs=output)
    """
    
    def __init__(self, inputs: Union[List, 'Input'], outputs: Union[List, Tensor]):
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        
        self.optimizer = None
        self.loss_fn = None
        self.metrics = []
        self.history = {'loss': [], 'val_loss': []}
        self.compiled = False
        
        # Build computation graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the computation graph from inputs to outputs."""
        # This is a simplified version - in practice, we'd trace through
        # the graph to find all layers and their connections
        self.layers = []
        self._layer_outputs = {}
        
        # For now, we'll assume layers are added in order
        # A full implementation would traverse the graph properly
    
    def compile(self, optimizer: Union[str, Optimizer], 
                loss: Union[str, Loss, Dict[str, Union[str, Loss]]],
                metrics: Optional[Union[List[str], Dict[str, List[str]]]] = None,
                loss_weights: Optional[Union[List[float], Dict[str, float]]] = None):
        """
        Compile the model for training.
        
        Parameters:
            optimizer: Optimizer to use
            loss: Loss function(s) to use
            metrics: Metrics to track
            loss_weights: Weights for different losses (multi-output models)
        """
        self.optimizer = get_optimizer(optimizer)
        
        # Handle loss functions
        if isinstance(loss, dict):
            self.loss_fn = {name: get_loss(l) for name, l in loss.items()}
        else:
            self.loss_fn = get_loss(loss)
        
        self.metrics = metrics or []
        self.loss_weights = loss_weights
        
        # Collect all parameters
        parameters = []
        for layer in self.layers:
            if hasattr(layer, 'get_parameters'):
                parameters.extend(layer.get_parameters())
        
        self.optimizer.add_parameters(parameters)
        self.compiled = True
    
    def forward(self, inputs: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        """Forward pass through the model."""
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # Simplified forward pass - full implementation would use the graph
        x = inputs[0]
        for layer in self.layers:
            x = layer(x)
        
        return x if len(self.outputs) == 1 else [x]
    
    def __call__(self, inputs: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        return self.forward(inputs)
    
    def fit(self, x: Union[Tensor, np.ndarray, List, Dict],
            y: Union[Tensor, np.ndarray, List, Dict],
            epochs: int = 10,
            batch_size: int = 32,
            validation_data: Optional[Tuple] = None,
            verbose: int = 1,
            callbacks: Optional[List] = None,
            shuffle: bool = True) -> dict:
        """
        Train the model.
        
        Parameters:
            x: Training data (can be list/dict for multi-input models)
            y: Training labels (can be list/dict for multi-output models)
            epochs: Number of epochs
            batch_size: Batch size
            validation_data: Validation data
            verbose: Verbosity mode
            callbacks: List of callbacks
            shuffle: Whether to shuffle data
        
        Returns:
            Training history
        """
        # This is a simplified implementation
        # Full implementation would handle multi-input/output properly
        
        # Convert to Sequential-like behavior for simplicity
        from .sequential import Sequential
        seq = Sequential(self.layers)
        seq.optimizer = self.optimizer
        seq.loss_fn = self.loss_fn
        seq.metrics = self.metrics
        seq.compiled = self.compiled
        
        return seq.fit(x, y, epochs, batch_size, validation_data, verbose, shuffle)
    
    def predict(self, x: Union[Tensor, np.ndarray, List, Dict],
                batch_size: int = 32) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate predictions."""
        # Simplified implementation
        from .sequential import Sequential
        seq = Sequential(self.layers)
        return seq.predict(x, batch_size)
    
    def evaluate(self, x: Union[Tensor, np.ndarray, List, Dict],
                 y: Union[Tensor, np.ndarray, List, Dict],
                 batch_size: int = 32) -> Dict[str, float]:
        """Evaluate the model."""
        # Simplified implementation
        from .sequential import Sequential
        seq = Sequential(self.layers)
        seq.loss_fn = self.loss_fn
        seq.metrics = self.metrics
        return seq.evaluate(x, y, batch_size)
    
    def summary(self):
        """Print model summary."""
        print("Model: Functional")
        print("=" * 65)
        print("Input shape(s):", [inp.shape for inp in self.inputs])
        print("Output shape(s):", [out.shape for out in self.outputs])
        print("=" * 65)
        
        # Would show full graph structure in complete implementation


class Input:
    """
    Input layer for functional API.
    
    Parameters:
        shape: Shape of input (excluding batch dimension)
        name: Optional name for the input
    """
    
    def __init__(self, shape: Tuple[int, ...], name: Optional[str] = None):
        self.shape = (None,) + shape  # Add batch dimension
        self.name = name or f"input_{id(self)}"
        self._tensor = None
    
    def __call__(self, x: Optional[Tensor] = None) -> Tensor:
        if x is None:
            # Create placeholder tensor
            self._tensor = Tensor.zeros(self.shape)
        else:
            self._tensor = x
        return self._tensor