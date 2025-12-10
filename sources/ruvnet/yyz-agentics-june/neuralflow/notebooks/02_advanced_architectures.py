"""
# Advanced Neural Network Architectures with NeuralFlow

This notebook demonstrates how to build various advanced neural network architectures
using NeuralFlow, including CNNs, RNNs, and custom architectures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import neuralflow as nf
from neuralflow.utils import data_utils, visualization

# %% [markdown]
# ## 1. Convolutional Neural Networks (CNNs)

# %%
print("=== Convolutional Neural Network ===\n")

# Create a CNN for image classification
cnn_model = nf.Sequential([
    # First convolutional block
    nf.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
    nf.layers.BatchNormalization(),
    nf.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
    nf.layers.MaxPool2D(pool_size=2),
    nf.layers.Dropout(0.25),
    
    # Second convolutional block
    nf.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    nf.layers.BatchNormalization(),
    nf.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    nf.layers.MaxPool2D(pool_size=2),
    nf.layers.Dropout(0.25),
    
    # Dense layers
    nf.layers.Dense(512, activation='relu'),
    nf.layers.Dropout(0.5),
    nf.layers.Dense(10, activation='softmax')
])

print("CNN Architecture:")
for i, layer in enumerate(cnn_model.layers):
    print(f"Layer {i+1}: {layer}")

# %% [markdown]
# ## 2. Recurrent Neural Networks (RNNs)

# %%
print("\n=== Recurrent Neural Network ===\n")

# Create an LSTM network for sequence processing
sequence_length = 50
vocab_size = 1000
embedding_dim = 128

lstm_model = nf.Sequential([
    # Embedding layer (simulated with Dense)
    nf.layers.Dense(embedding_dim, activation='linear'),
    
    # LSTM layers
    nf.layers.LSTM(256, return_sequences=True),
    nf.layers.Dropout(0.3),
    nf.layers.LSTM(128, return_sequences=False),
    nf.layers.Dropout(0.3),
    
    # Output layer
    nf.layers.Dense(vocab_size, activation='softmax')
])

print("LSTM Architecture:")
for i, layer in enumerate(lstm_model.layers):
    print(f"Layer {i+1}: {layer}")

# %% [markdown]
# ## 3. Deep Residual Networks (ResNet-style)

# %%
print("\n=== Residual Network Block ===\n")

def residual_block(x, filters, kernel_size=3):
    """Create a residual block."""
    # Save input for skip connection
    shortcut = x
    
    # First convolution
    x = nf.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = nf.layers.BatchNormalization()(x)
    
    # Second convolution
    x = nf.layers.Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    x = nf.layers.BatchNormalization()(x)
    
    # Add skip connection (conceptual - would need Add layer)
    # x = Add()([x, shortcut])
    # x = Activation('relu')(x)
    
    return x

print("Residual block structure defined")
print("Components: Conv2D -> BN -> ReLU -> Conv2D -> BN -> Add -> ReLU")

# %% [markdown]
# ## 4. Autoencoder Architecture

# %%
print("\n=== Autoencoder ===\n")

# Create an autoencoder for dimensionality reduction
input_dim = 784  # e.g., flattened 28x28 images
encoding_dim = 32

# Encoder
encoder = nf.Sequential([
    nf.layers.Dense(256, activation='relu'),
    nf.layers.Dense(128, activation='relu'),
    nf.layers.Dense(64, activation='relu'),
    nf.layers.Dense(encoding_dim, activation='relu')  # Bottleneck
])

# Decoder
decoder = nf.Sequential([
    nf.layers.Dense(64, activation='relu'),
    nf.layers.Dense(128, activation='relu'),
    nf.layers.Dense(256, activation='relu'),
    nf.layers.Dense(input_dim, activation='sigmoid')  # Reconstruction
])

# Full autoencoder (conceptual)
print("Encoder Architecture:")
for i, layer in enumerate(encoder.layers):
    print(f"  Layer {i+1}: {layer}")

print("\nDecoder Architecture:")
for i, layer in enumerate(decoder.layers):
    print(f"  Layer {i+1}: {layer}")

# %% [markdown]
# ## 5. Multi-Input/Multi-Output Networks (Functional API Style)

# %%
print("\n=== Multi-Input Network (Conceptual) ===\n")

# Example: Network with image and text inputs
print("Multi-modal network structure:")
print("1. Image Input -> CNN Branch")
print("   - Conv2D(32) -> MaxPool2D -> Conv2D(64) -> GlobalAvgPool2D")
print("2. Text Input -> RNN Branch")
print("   - Embedding -> LSTM(128) -> LSTM(64)")
print("3. Merge branches -> Concatenate")
print("4. Combined features -> Dense(256) -> Dense(128) -> Output")

# %% [markdown]
# ## 6. Custom Layer Example

# %%
print("\n=== Custom Layer Implementation ===\n")

class CustomActivation:
    """Custom activation layer example."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.name = f"custom_activation_{id(self)}"
    
    def forward(self, x):
        """Custom activation: alpha * x * sigmoid(x)"""
        from neuralflow.core.activations import sigmoid
        return self.alpha * x * sigmoid(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def get_parameters(self):
        return []  # No trainable parameters

print("Custom activation layer defined: f(x) = α * x * sigmoid(x)")

# Example usage
custom_model = nf.Sequential([
    nf.layers.Dense(64),
    CustomActivation(alpha=2.0),
    nf.layers.Dense(32),
    CustomActivation(alpha=1.5),
    nf.layers.Dense(10, activation='softmax')
])

print("\nModel with custom activation:")
for i, layer in enumerate(custom_model.layers):
    print(f"Layer {i+1}: {layer}")

# %% [markdown]
# ## 7. Training Strategies and Tips

# %%
print("\n=== Advanced Training Strategies ===\n")

# Learning rate scheduling
print("1. Learning Rate Scheduling:")
print("   - Start with higher learning rate (e.g., 0.01)")
print("   - Reduce on plateau or exponentially")
print("   - Example schedule: lr = lr * 0.95 every 5 epochs")

# Regularization techniques
print("\n2. Regularization Techniques:")
print("   - Dropout: Already implemented in layers")
print("   - Batch Normalization: Normalizes activations")
print("   - L2 regularization: Add penalty to weights")
print("   - Data augmentation: Transform training data")

# Advanced optimizers
print("\n3. Advanced Optimizers:")
print("   - Adam: Adaptive learning rates")
print("   - RMSprop: Good for RNNs")
print("   - SGD with momentum: Classic choice")

# %% [markdown]
# ## 8. Performance Optimization

# %%
print("\n=== Performance Tips ===\n")

print("1. Batch Processing:")
print("   - Use appropriate batch sizes (32, 64, 128)")
print("   - Larger batches = faster training but more memory")

print("\n2. Model Architecture:")
print("   - Start simple, add complexity gradually")
print("   - Use batch normalization for deeper networks")
print("   - Consider skip connections for very deep models")

print("\n3. Data Preprocessing:")
print("   - Normalize inputs to [-1, 1] or [0, 1]")
print("   - Use data augmentation for small datasets")
print("   - Balance classes for classification")

# %% [markdown]
# ## Example: Training a Complex Model

# %%
print("\n=== Training Example ===\n")

# Generate synthetic data
X = np.random.randn(1000, 28, 28, 1)
y = data_utils.to_categorical(np.random.randint(0, 10, 1000), 10)

# Create and compile model
model = nf.Sequential([
    nf.layers.Conv2D(32, 3, padding='same', activation='relu'),
    nf.layers.MaxPool2D(2),
    nf.layers.Dense(128, activation='relu'),
    nf.layers.Dropout(0.5),
    nf.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=nf.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model ready for training!")
print("\nTo train:")
print("```python")
print("history = model.fit(X_train, y_train,")
print("                    epochs=20,")
print("                    batch_size=32,")
print("                    validation_split=0.2)")
print("```")

# %% [markdown]
# ## Summary
# 
# This notebook covered:
# - CNN architectures for image processing
# - RNN/LSTM networks for sequences
# - Residual networks concepts
# - Autoencoders for unsupervised learning
# - Multi-input/output architectures
# - Custom layer implementation
# - Advanced training strategies
# - Performance optimization tips
# 
# NeuralFlow provides flexibility to build various architectures while maintaining simplicity.

print("\nNotebook completed successfully!")