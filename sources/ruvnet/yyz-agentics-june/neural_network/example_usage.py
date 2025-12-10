"""
Example usage of the neural network components.
Demonstrates how to use layers, activations, and initializers.
"""

import numpy as np
import sys
sys.path.append('/workspaces/claude-test')

from neural_network.core import (
    Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization,
    ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, ELU, Swish, GELU,
    get_activation, get_initializer
)


def example_dense_network():
    """Example of building a simple dense neural network."""
    print("=== Dense Network Example ===")
    
    # Create layers
    dense1 = Dense(128, activation=ReLU(), kernel_initializer='he_normal')
    dropout1 = Dropout(0.2)
    dense2 = Dense(64, activation=ReLU(), kernel_initializer='he_normal')
    dense3 = Dense(10, activation=Softmax())
    
    # Create sample input
    batch_size = 32
    input_dim = 784  # e.g., flattened 28x28 image
    x = np.random.randn(batch_size, input_dim)
    
    # Forward pass
    h1 = dense1.forward(x, training=True)
    h1_drop = dropout1.forward(h1, training=True)
    h2 = dense2.forward(h1_drop, training=True)
    output = dense3.forward(h2, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Dense1 output shape: {h1.shape}")
    print(f"Dense2 output shape: {h2.shape}")
    print(f"Final output shape: {output.shape}")
    print(f"Output sum per sample: {output.sum(axis=1)[:5]}")  # Should be ~1.0
    
    # Backward pass (simulate gradient from loss)
    grad_output = np.random.randn(*output.shape)
    
    grad_h2 = dense3.backward(grad_output)
    grad_h1_drop = dense2.backward(grad_h2)
    grad_h1 = dropout1.backward(grad_h1_drop)
    grad_input = dense1.backward(grad_h1)
    
    print(f"\nGradient shapes:")
    print(f"Input gradient shape: {grad_input.shape}")
    print(f"Dense1 weight gradient shape: {dense1.grads['W'].shape}")
    print(f"Dense1 bias gradient shape: {dense1.grads['b'].shape}")
    

def example_conv_network():
    """Example of building a convolutional neural network."""
    print("\n\n=== Convolutional Network Example ===")
    
    # Create layers
    conv1 = Conv2D(32, kernel_size=3, padding='same', activation=ReLU())
    batchnorm1 = BatchNormalization()
    maxpool1 = MaxPool2D(pool_size=2)
    
    conv2 = Conv2D(64, kernel_size=3, padding='same', activation=ReLU())
    batchnorm2 = BatchNormalization()
    maxpool2 = MaxPool2D(pool_size=2)
    
    # Create sample input (batch_size, height, width, channels)
    batch_size = 16
    x = np.random.randn(batch_size, 28, 28, 1)  # e.g., MNIST
    
    # Forward pass
    h1 = conv1.forward(x, training=True)
    h1_bn = batchnorm1.forward(h1, training=True)
    h1_pool = maxpool1.forward(h1_bn, training=True)
    
    h2 = conv2.forward(h1_pool, training=True)
    h2_bn = batchnorm2.forward(h2, training=True)
    h2_pool = maxpool2.forward(h2_bn, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"After Conv1 + BN: {h1_bn.shape}")
    print(f"After MaxPool1: {h1_pool.shape}")
    print(f"After Conv2 + BN: {h2_bn.shape}")
    print(f"After MaxPool2: {h2_pool.shape}")
    
    # Backward pass
    grad_output = np.random.randn(*h2_pool.shape)
    
    grad_h2_bn = maxpool2.backward(grad_output)
    grad_h2 = batchnorm2.backward(grad_h2_bn)
    grad_h1_pool = conv2.backward(grad_h2)
    
    grad_h1_bn = maxpool1.backward(grad_h1_pool)
    grad_h1 = batchnorm1.backward(grad_h1_bn)
    grad_input = conv1.backward(grad_h1)
    
    print(f"\nGradient shapes:")
    print(f"Input gradient shape: {grad_input.shape}")
    print(f"Conv1 kernel gradient shape: {conv1.grads['W'].shape}")
    print(f"Conv2 kernel gradient shape: {conv2.grads['W'].shape}")


def example_initializers():
    """Example of using different weight initializers."""
    print("\n\n=== Weight Initializers Example ===")
    
    shape = (100, 50)
    
    # Test different initializers
    initializers = {
        'xavier_normal': get_initializer('xavier_normal'),
        'xavier_uniform': get_initializer('xavier_uniform'),
        'he_normal': get_initializer('he_normal'),
        'he_uniform': get_initializer('he_uniform'),
        'random_normal': get_initializer('random_normal'),
        'zeros': get_initializer('zeros'),
        'ones': get_initializer('ones')
    }
    
    for name, init in initializers.items():
        weights = init(shape)
        print(f"\n{name}:")
        print(f"  Shape: {weights.shape}")
        print(f"  Mean: {weights.mean():.4f}")
        print(f"  Std: {weights.std():.4f}")
        print(f"  Min: {weights.min():.4f}")
        print(f"  Max: {weights.max():.4f}")


def example_activations():
    """Example of using different activation functions."""
    print("\n\n=== Activation Functions Example ===")
    
    # Create sample input
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    
    activations = {
        'ReLU': ReLU(),
        'LeakyReLU': LeakyReLU(alpha=0.1),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'ELU': ELU(alpha=1.0),
        'Swish': Swish(),
        'GELU': GELU()
    }
    
    print(f"Input: {x}")
    print("\nForward pass:")
    
    for name, activation in activations.items():
        output = activation.forward(x)
        print(f"{name}: {output}")
        
    # Test softmax separately (needs 2D input)
    print("\nSoftmax example:")
    x_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    softmax = Softmax()
    output = softmax.forward(x_2d)
    print(f"Input:\n{x_2d}")
    print(f"Output:\n{output}")
    print(f"Sum per row: {output.sum(axis=1)}")


def example_spatial_dropout():
    """Example of spatial dropout for CNNs."""
    print("\n\n=== Spatial Dropout Example ===")
    
    from neural_network.core import SpatialDropout2D
    
    # Create layer
    spatial_dropout = SpatialDropout2D(rate=0.2)
    
    # Create sample feature maps
    batch_size = 4
    height, width = 8, 8
    channels = 16
    x = np.ones((batch_size, height, width, channels))
    
    # Apply spatial dropout
    output = spatial_dropout.forward(x, training=True)
    
    # Check which channels were dropped (should be same across spatial dimensions)
    dropped_channels = (output[0, 0, 0, :] == 0)
    print(f"Input shape: {x.shape}")
    print(f"Number of channels dropped: {dropped_channels.sum()} out of {channels}")
    print(f"Dropped channels are consistent across spatial dims: "
          f"{np.all(output[0, :, :, dropped_channels] == 0)}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    example_dense_network()
    example_conv_network()
    example_initializers()
    example_activations()
    example_spatial_dropout()
    
    print("\n\nAll examples completed successfully!")