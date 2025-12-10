# NeuralFlow

<div align="center">

**A Comprehensive Neural Network Library for Rapid Prototyping and Deployment**

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://neuralflow.readthedocs.io)

</div>

## Overview

NeuralFlow is a flexible and easy-to-use neural network library designed for rapid prototyping and deployment of deep learning models. Built with simplicity and extensibility in mind, it provides a clean API for building, training, and deploying neural networks.

## Features

- **Simple and Intuitive API**: Build complex models with just a few lines of code
- **Comprehensive Layer Support**: Dense, Conv2D, LSTM, GRU, BatchNormalization, and more
- **Multiple Optimizers**: SGD, Adam, RMSprop, AdaGrad with customizable parameters
- **Flexible Architecture**: Support for Sequential and Functional API styles
- **Built-in Data Utilities**: Data preprocessing, augmentation, and batch generation
- **Visualization Tools**: Training history, confusion matrices, and model architecture visualization
- **Pure Python Implementation**: Easy to understand and modify

## Installation

### From PyPI (Recommended)
```bash
pip install neuralflow
```

### From Source
```bash
git clone https://github.com/neuralflow/neuralflow.git
cd neuralflow
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev,notebooks,visualization]"
```

## Quick Start

### Basic Classification Example

```python
import neuralflow as nf
from neuralflow.utils import data_utils

# Create a simple neural network
model = nf.Sequential([
    nf.layers.Dense(128, activation='relu'),
    nf.layers.Dropout(0.5),
    nf.layers.Dense(64, activation='relu'),
    nf.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2%}')
```

### Convolutional Neural Network

```python
# Build a CNN for image classification
cnn = nf.Sequential([
    nf.layers.Conv2D(32, kernel_size=3, activation='relu'),
    nf.layers.MaxPool2D(pool_size=2),
    nf.layers.Conv2D(64, kernel_size=3, activation='relu'),
    nf.layers.MaxPool2D(pool_size=2),
    nf.layers.Dense(128, activation='relu'),
    nf.layers.Dropout(0.5),
    nf.layers.Dense(10, activation='softmax')
])
```

### Recurrent Neural Network

```python
# Build an LSTM for sequence processing
lstm_model = nf.Sequential([
    nf.layers.LSTM(128, return_sequences=True),
    nf.layers.LSTM(64),
    nf.layers.Dense(10, activation='softmax')
])
```

## Core Components

### Layers
- **Dense**: Fully connected layer
- **Conv2D**: 2D convolutional layer
- **MaxPool2D**: Max pooling layer
- **LSTM**: Long Short-Term Memory layer
- **GRU**: Gated Recurrent Unit layer
- **Dropout**: Regularization layer
- **BatchNormalization**: Normalization layer
- **LayerNormalization**: Layer-wise normalization

### Activations
- ReLU, LeakyReLU, Sigmoid, Tanh, Softmax

### Optimizers
- SGD (with momentum and Nesterov)
- Adam
- RMSprop
- AdaGrad

### Loss Functions
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Binary Crossentropy
- Categorical Crossentropy

## Demos and Examples

The library includes several demonstration applications:

1. **Image Classification Demo**: CNN and dense networks for image classification
2. **Text Generation Demo**: RNN-based text generation
3. **Regression Demo**: Neural networks for regression tasks
4. **Binary Classification Demo**: Circle dataset classification
5. **Multi-class Classification Demo**: Spiral dataset classification

Run demos:
```bash
python -m neuralflow.demos.image_classification_demo
python -m neuralflow.demos.text_generation_demo
python -m neuralflow.demos.regression_classification_demo
```

## Example Notebooks

Interactive notebooks are provided to demonstrate various aspects of the library:

1. `01_getting_started.py`: Introduction to NeuralFlow basics
2. `02_advanced_architectures.py`: Complex network architectures
3. `03_practical_examples.py`: Real-world inspired examples

Convert to Jupyter notebooks:
```bash
jupytext --to notebook neuralflow/notebooks/01_getting_started.py
```

## Documentation

For detailed documentation, visit [https://neuralflow.readthedocs.io](https://neuralflow.readthedocs.io)

### API Reference

```python
# Sequential Model
model = nf.Sequential(layers)
model.compile(optimizer, loss, metrics)
model.fit(X, y, epochs, batch_size, validation_data)
model.evaluate(X, y)
model.predict(X)

# Layers
layer = nf.layers.Dense(units, activation)
layer = nf.layers.Conv2D(filters, kernel_size, padding, activation)
layer = nf.layers.LSTM(units, return_sequences)

# Optimizers
optimizer = nf.optimizers.Adam(learning_rate)
optimizer = nf.optimizers.SGD(learning_rate, momentum)

# Losses
loss = nf.losses.categorical_crossentropy
loss = nf.losses.mean_squared_error
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/neuralflow/neuralflow.git
cd neuralflow

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black neuralflow/

# Check code style
flake8 neuralflow/
```

## Roadmap

- [ ] GPU acceleration support
- [ ] More layer types (ConvLSTM, Attention, Transformer)
- [ ] Model serialization and loading
- [ ] Distributed training support
- [ ] Mobile deployment tools
- [ ] AutoML capabilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use NeuralFlow in your research, please cite:

```bibtex
@software{neuralflow2024,
  title = {NeuralFlow: A Comprehensive Neural Network Library},
  author = {NeuralFlow Team},
  year = {2024},
  url = {https://github.com/neuralflow/neuralflow}
}
```

## Acknowledgments

- Inspired by Keras, PyTorch, and TensorFlow
- Built for educational purposes and rapid prototyping
- Special thanks to all contributors

---

<div align="center">
Made with ❤️ by the NeuralFlow Team
</div>