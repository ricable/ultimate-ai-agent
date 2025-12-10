"""
# Getting Started with NeuralFlow

This notebook introduces the basics of using NeuralFlow for building and training neural networks.

## Installation

```bash
pip install -e /path/to/neuralflow
```

## Basic Concepts

NeuralFlow provides a simple and intuitive API for building neural networks. The main components are:

1. **Layers**: Building blocks of neural networks (Dense, Conv2D, LSTM, etc.)
2. **Models**: Containers for layers (Sequential, Model)
3. **Optimizers**: Training algorithms (SGD, Adam, RMSprop)
4. **Loss Functions**: Objectives to minimize (MSE, CrossEntropy)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import neuralflow as nf
from neuralflow.utils import data_utils, visualization

print("NeuralFlow version:", nf.__version__)

# %% [markdown]
# ## 1. Building Your First Model

# %%
# Create a simple neural network
model = nf.Sequential([
    nf.layers.Dense(128, activation='relu'),
    nf.layers.Dropout(0.5),
    nf.layers.Dense(64, activation='relu'),
    nf.layers.Dense(10, activation='softmax')
])

print("Model created successfully!")

# %% [markdown]
# ## 2. Preparing Data

# %%
# Generate synthetic data for demonstration
n_samples = 1000
n_features = 20
n_classes = 10

# Random features
X = np.random.randn(n_samples, n_features)

# Random labels
y = np.random.randint(0, n_classes, n_samples)
y_categorical = data_utils.to_categorical(y, n_classes)

# Split into train and test
X_train, X_test, y_train, y_test = data_utils.train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape}")
print(f"Test samples: {X_test.shape}")

# %% [markdown]
# ## 3. Compiling the Model

# %%
# Compile the model with optimizer and loss
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled!")

# %% [markdown]
# ## 4. Training the Model

# %%
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# %% [markdown]
# ## 5. Evaluating Performance

# %%
# Evaluate on test set
test_metrics = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {test_metrics['loss']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")

# Visualize training history
print("\nTraining History:")
print(visualization.plot_history(history))

# %% [markdown]
# ## 6. Making Predictions

# %%
# Make predictions on new data
new_data = np.random.randn(5, n_features)
predictions = model.predict(new_data)
predicted_classes = np.argmax(predictions, axis=-1)

print("\nPredictions for 5 new samples:")
for i, (pred_class, probs) in enumerate(zip(predicted_classes, predictions)):
    confidence = probs[pred_class]
    print(f"Sample {i+1}: Class {pred_class} (confidence: {confidence:.2%})")

# %% [markdown]
# ## 7. Saving and Loading Models (Conceptual)

# %%
print("\nTo save a model:")
print("```python")
print("# Save model architecture and weights")
print("model.save('my_model.h5')")
print("")
print("# Load model")
print("loaded_model = nf.models.load_model('my_model.h5')")
print("```")

# %% [markdown]
# ## Summary
# 
# In this notebook, we covered:
# - Creating a neural network model
# - Preparing and splitting data
# - Compiling the model with optimizer and loss
# - Training the model
# - Evaluating performance
# - Making predictions
# 
# Next steps:
# - Try different architectures
# - Experiment with hyperparameters
# - Work with real datasets
# - Explore advanced features like custom layers and callbacks

print("\nNotebook completed successfully!")