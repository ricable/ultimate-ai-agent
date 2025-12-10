"""
Recurrent Neural Network Layers
"""
import numpy as np
from typing import Optional, Tuple, Union
from ..tensor import Tensor
from ..activations import sigmoid, tanh


class LSTM:
    """
    Long Short-Term Memory (LSTM) layer.
    
    Parameters:
        units: Number of LSTM units
        return_sequences: Whether to return full sequence or just last output
        return_state: Whether to return the final state
    """
    
    def __init__(self,
                 units: int,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 weight_initializer: str = 'glorot_uniform',
                 name: Optional[str] = None):
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.weight_initializer = weight_initializer
        self.name = name or f"lstm_{id(self)}"
        
        # Parameters
        self.W_i = None  # Input gate weights
        self.W_f = None  # Forget gate weights
        self.W_c = None  # Cell gate weights
        self.W_o = None  # Output gate weights
        
        self.U_i = None  # Input gate recurrent weights
        self.U_f = None  # Forget gate recurrent weights
        self.U_c = None  # Cell gate recurrent weights
        self.U_o = None  # Output gate recurrent weights
        
        self.b_i = None  # Input gate bias
        self.b_f = None  # Forget gate bias
        self.b_c = None  # Cell gate bias
        self.b_o = None  # Output gate bias
        
        self.built = False
    
    def build(self, input_shape: tuple):
        """Initialize weights based on input shape."""
        # input_shape: (batch_size, timesteps, features)
        input_dim = input_shape[-1]
        
        # Initialize weights
        if self.weight_initializer == 'glorot_uniform':
            limit_w = np.sqrt(6 / (input_dim + self.units))
            limit_u = np.sqrt(6 / (self.units + self.units))
        else:
            limit_w = limit_u = 0.05
        
        # Input weights
        self.W_i = Tensor.uniform((input_dim, self.units), -limit_w, limit_w, requires_grad=True)
        self.W_f = Tensor.uniform((input_dim, self.units), -limit_w, limit_w, requires_grad=True)
        self.W_c = Tensor.uniform((input_dim, self.units), -limit_w, limit_w, requires_grad=True)
        self.W_o = Tensor.uniform((input_dim, self.units), -limit_w, limit_w, requires_grad=True)
        
        # Recurrent weights
        self.U_i = Tensor.uniform((self.units, self.units), -limit_u, limit_u, requires_grad=True)
        self.U_f = Tensor.uniform((self.units, self.units), -limit_u, limit_u, requires_grad=True)
        self.U_c = Tensor.uniform((self.units, self.units), -limit_u, limit_u, requires_grad=True)
        self.U_o = Tensor.uniform((self.units, self.units), -limit_u, limit_u, requires_grad=True)
        
        # Biases
        self.b_i = Tensor.zeros((self.units,), requires_grad=True)
        self.b_f = Tensor.ones((self.units,), requires_grad=True)  # Initialize forget gate bias to 1
        self.b_c = Tensor.zeros((self.units,), requires_grad=True)
        self.b_o = Tensor.zeros((self.units,), requires_grad=True)
        
        self.built = True
    
    def forward(self, x: Tensor, initial_state: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Forward pass through LSTM."""
        if not self.built:
            self.build(x.shape)
        
        batch_size, timesteps, _ = x.shape
        
        # Initialize hidden state and cell state
        if initial_state is None:
            h_t = Tensor.zeros((batch_size, self.units), requires_grad=True)
            c_t = Tensor.zeros((batch_size, self.units), requires_grad=True)
        else:
            h_t, c_t = initial_state
        
        outputs = []
        
        # Process each timestep
        for t in range(timesteps):
            x_t = Tensor(x.data[:, t, :], requires_grad=x.requires_grad)
            
            # Input gate
            i_t = sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            
            # Forget gate
            f_t = sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            
            # Cell gate
            c_tilde = tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            
            # Update cell state
            c_t = f_t * c_t + i_t * c_tilde
            
            # Output gate
            o_t = sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            
            # Update hidden state
            h_t = o_t * tanh(c_t)
            
            outputs.append(h_t)
        
        # Prepare output
        if self.return_sequences:
            # Stack all outputs
            output = Tensor(np.stack([o.data for o in outputs], axis=1), requires_grad=True)
        else:
            # Return only last output
            output = outputs[-1]
        
        if self.return_state:
            return output, h_t, c_t
        else:
            return output
    
    def __call__(self, x: Tensor, initial_state: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, ...]]:
        return self.forward(x, initial_state)
    
    def get_parameters(self) -> list:
        """Get all trainable parameters."""
        return [
            self.W_i, self.W_f, self.W_c, self.W_o,
            self.U_i, self.U_f, self.U_c, self.U_o,
            self.b_i, self.b_f, self.b_c, self.b_o
        ]


class GRU:
    """
    Gated Recurrent Unit (GRU) layer.
    
    Parameters:
        units: Number of GRU units
        return_sequences: Whether to return full sequence or just last output
        return_state: Whether to return the final state
    """
    
    def __init__(self,
                 units: int,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 weight_initializer: str = 'glorot_uniform',
                 name: Optional[str] = None):
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.weight_initializer = weight_initializer
        self.name = name or f"gru_{id(self)}"
        
        # Parameters
        self.W_z = None  # Update gate weights
        self.W_r = None  # Reset gate weights
        self.W_h = None  # Candidate weights
        
        self.U_z = None  # Update gate recurrent weights
        self.U_r = None  # Reset gate recurrent weights
        self.U_h = None  # Candidate recurrent weights
        
        self.b_z = None  # Update gate bias
        self.b_r = None  # Reset gate bias
        self.b_h = None  # Candidate bias
        
        self.built = False
    
    def build(self, input_shape: tuple):
        """Initialize weights based on input shape."""
        # input_shape: (batch_size, timesteps, features)
        input_dim = input_shape[-1]
        
        # Initialize weights
        if self.weight_initializer == 'glorot_uniform':
            limit_w = np.sqrt(6 / (input_dim + self.units))
            limit_u = np.sqrt(6 / (self.units + self.units))
        else:
            limit_w = limit_u = 0.05
        
        # Input weights
        self.W_z = Tensor.uniform((input_dim, self.units), -limit_w, limit_w, requires_grad=True)
        self.W_r = Tensor.uniform((input_dim, self.units), -limit_w, limit_w, requires_grad=True)
        self.W_h = Tensor.uniform((input_dim, self.units), -limit_w, limit_w, requires_grad=True)
        
        # Recurrent weights
        self.U_z = Tensor.uniform((self.units, self.units), -limit_u, limit_u, requires_grad=True)
        self.U_r = Tensor.uniform((self.units, self.units), -limit_u, limit_u, requires_grad=True)
        self.U_h = Tensor.uniform((self.units, self.units), -limit_u, limit_u, requires_grad=True)
        
        # Biases
        self.b_z = Tensor.zeros((self.units,), requires_grad=True)
        self.b_r = Tensor.zeros((self.units,), requires_grad=True)
        self.b_h = Tensor.zeros((self.units,), requires_grad=True)
        
        self.built = True
    
    def forward(self, x: Tensor, initial_state: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through GRU."""
        if not self.built:
            self.build(x.shape)
        
        batch_size, timesteps, _ = x.shape
        
        # Initialize hidden state
        if initial_state is None:
            h_t = Tensor.zeros((batch_size, self.units), requires_grad=True)
        else:
            h_t = initial_state
        
        outputs = []
        
        # Process each timestep
        for t in range(timesteps):
            x_t = Tensor(x.data[:, t, :], requires_grad=x.requires_grad)
            
            # Update gate
            z_t = sigmoid(x_t @ self.W_z + h_t @ self.U_z + self.b_z)
            
            # Reset gate
            r_t = sigmoid(x_t @ self.W_r + h_t @ self.U_r + self.b_r)
            
            # Candidate hidden state
            h_tilde = tanh(x_t @ self.W_h + (r_t * h_t) @ self.U_h + self.b_h)
            
            # Update hidden state
            h_t = z_t * h_t + (Tensor.ones(z_t.shape) - z_t) * h_tilde
            
            outputs.append(h_t)
        
        # Prepare output
        if self.return_sequences:
            # Stack all outputs
            output = Tensor(np.stack([o.data for o in outputs], axis=1), requires_grad=True)
        else:
            # Return only last output
            output = outputs[-1]
        
        if self.return_state:
            return output, h_t
        else:
            return output
    
    def __call__(self, x: Tensor, initial_state: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self.forward(x, initial_state)
    
    def get_parameters(self) -> list:
        """Get all trainable parameters."""
        return [
            self.W_z, self.W_r, self.W_h,
            self.U_z, self.U_r, self.U_h,
            self.b_z, self.b_r, self.b_h
        ]