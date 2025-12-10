"""
Dropout Layer for regularization
"""
import numpy as np
from typing import Optional
from ..tensor import Tensor


class Dropout:
    """
    Dropout layer for regularization.
    
    Parameters:
        rate: Dropout rate (fraction of units to drop)
        seed: Random seed for reproducibility
    """
    
    def __init__(self, rate: float = 0.5, seed: Optional[int] = None, name: Optional[str] = None):
        if not 0 <= rate < 1:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
        
        self.rate = rate
        self.seed = seed
        self.name = name or f"dropout_{id(self)}"
        self.training = True
        self._rng = np.random.RandomState(seed)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through dropout layer."""
        if not self.training or self.rate == 0:
            return x
        
        # Generate dropout mask
        keep_prob = 1 - self.rate
        mask = self._rng.binomial(1, keep_prob, size=x.shape) / keep_prob
        
        output_data = x.data * mask
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        def _backward(grad):
            if x.requires_grad:
                x.backward(grad * mask)
        
        if output.requires_grad:
            output._grad_fn = _backward
            output._prev = {x}
        
        return output
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def train(self):
        """Set layer to training mode."""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False
    
    def get_parameters(self) -> list:
        """Dropout has no trainable parameters."""
        return []