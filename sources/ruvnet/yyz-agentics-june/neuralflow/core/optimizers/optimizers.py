"""
Optimizer implementations
"""
import numpy as np
from typing import List, Union, Optional
from ..tensor import Tensor


class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.parameters = []
    
    def add_parameters(self, parameters: List[Tensor]):
        """Add parameters to optimize."""
        self.parameters.extend(parameters)
    
    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            if param is not None:
                param.zero_grad()
    
    def step(self):
        """Update parameters based on gradients."""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum.
    
    Parameters:
        learning_rate: Learning rate
        momentum: Momentum factor
        nesterov: Whether to use Nesterov momentum
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}
    
    def step(self):
        """Update parameters using SGD."""
        for param in self.parameters:
            if param is None or param.grad is None:
                continue
            
            if self.momentum != 0:
                # Get or initialize velocity
                if id(param) not in self.velocities:
                    self.velocities[id(param)] = np.zeros_like(param.data)
                
                velocity = self.velocities[id(param)]
                
                # Update velocity
                velocity = self.momentum * velocity - self.learning_rate * param.grad
                self.velocities[id(param)] = velocity
                
                # Update parameters
                if self.nesterov:
                    param.data += self.momentum * velocity - self.learning_rate * param.grad
                else:
                    param.data += velocity
            else:
                # Standard gradient descent
                param.data -= self.learning_rate * param.grad


class Adam(Optimizer):
    """
    Adam optimizer.
    
    Parameters:
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def step(self):
        """Update parameters using Adam."""
        self.t += 1
        
        for param in self.parameters:
            if param is None or param.grad is None:
                continue
            
            # Initialize moments if needed
            if id(param) not in self.m:
                self.m[id(param)] = np.zeros_like(param.data)
                self.v[id(param)] = np.zeros_like(param.data)
            
            # Get moments
            m = self.m[id(param)]
            v = self.v[id(param)]
            
            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * param.grad
            
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (param.grad ** 2)
            
            # Store updated moments
            self.m[id(param)] = m
            self.v[id(param)] = v
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Parameters:
        learning_rate: Learning rate
        rho: Decay rate
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}
    
    def step(self):
        """Update parameters using RMSprop."""
        for param in self.parameters:
            if param is None or param.grad is None:
                continue
            
            # Initialize cache if needed
            if id(param) not in self.cache:
                self.cache[id(param)] = np.zeros_like(param.data)
            
            # Update cache
            cache = self.cache[id(param)]
            cache = self.rho * cache + (1 - self.rho) * (param.grad ** 2)
            self.cache[id(param)] = cache
            
            # Update parameters
            param.data -= self.learning_rate * param.grad / (np.sqrt(cache) + self.epsilon)


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.
    
    Parameters:
        learning_rate: Learning rate
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}
    
    def step(self):
        """Update parameters using AdaGrad."""
        for param in self.parameters:
            if param is None or param.grad is None:
                continue
            
            # Initialize cache if needed
            if id(param) not in self.cache:
                self.cache[id(param)] = np.zeros_like(param.data)
            
            # Update cache
            self.cache[id(param)] += param.grad ** 2
            
            # Update parameters
            param.data -= self.learning_rate * param.grad / (np.sqrt(self.cache[id(param)]) + self.epsilon)


# Helper function to get optimizer by name
def get_optimizer(optimizer: Union[str, Optimizer], learning_rate: Optional[float] = None) -> Optimizer:
    """Get optimizer by name or return the optimizer if already instantiated."""
    if isinstance(optimizer, Optimizer):
        return optimizer
    
    optimizer_map = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSprop,
        'adagrad': AdaGrad
    }
    
    if optimizer.lower() in optimizer_map:
        opt_class = optimizer_map[optimizer.lower()]
        if learning_rate is not None:
            return opt_class(learning_rate=learning_rate)
        else:
            return opt_class()
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")