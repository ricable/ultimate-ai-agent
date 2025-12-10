# Neural Network Components - Pure NumPy Implementation

A comprehensive collection of neural network components implemented using only NumPy for CPU optimization. This implementation includes all essential layers, activation functions, and weight initializers with full forward and backward propagation support.

## Features

### Core Layers
- **Dense (Fully Connected)**: Standard feed-forward layer with configurable units
- **Conv2D**: 2D Convolutional layer with padding and stride support
- **MaxPool2D**: Max pooling layer for spatial downsampling
- **AveragePool2D**: Average pooling layer
- **Dropout**: Standard dropout for regularization
- **SpatialDropout2D**: Drops entire feature maps for CNNs
- **AlphaDropout**: Self-normalizing dropout for SELU networks
- **BatchNormalization**: Batch normalization with moving statistics

### Activation Functions
- **ReLU**: Rectified Linear Unit
- **LeakyReLU**: Leaky ReLU with configurable alpha
- **Sigmoid**: Logistic sigmoid function
- **Tanh**: Hyperbolic tangent
- **Softmax**: Multi-class probability distribution
- **ELU**: Exponential Linear Unit
- **Swish**: Self-gated activation (x * sigmoid(x))
- **GELU**: Gaussian Error Linear Unit

### Weight Initializers
- **Xavier/Glorot Normal & Uniform**: For networks with sigmoid/tanh
- **He Normal & Uniform**: Optimized for ReLU networks
- **Random Normal & Uniform**: Basic random initialization
- **Zeros & Ones**: Constant initializers

## Installation

Simply clone or copy the `neural_network` directory to your project:

```bash
cp -r neural_network /path/to/your/project/
```

## Usage

### Basic Dense Network

```python
from neural_network.core import Dense, ReLU, Softmax

# Create layers
layer1 = Dense(128, activation=ReLU(), kernel_initializer='he_normal')
layer2 = Dense(10, activation=Softmax())

# Forward pass
x = np.random.randn(32, 784)  # batch_size=32, input_dim=784
h1 = layer1.forward(x)
output = layer2.forward(h1)

# Backward pass
grad_output = compute_loss_gradient(output, targets)
grad_h1 = layer2.backward(grad_output)
grad_input = layer1.backward(grad_h1)
```

### Convolutional Network

```python
from neural_network.core import Conv2D, MaxPool2D, BatchNormalization, ReLU

# Create CNN layers
conv1 = Conv2D(32, kernel_size=3, padding='same', activation=ReLU())
bn1 = BatchNormalization()
pool1 = MaxPool2D(pool_size=2)

# Forward pass
x = np.random.randn(16, 28, 28, 1)  # MNIST-like input
h1 = conv1.forward(x)
h1_bn = bn1.forward(h1, training=True)
h1_pool = pool1.forward(h1_bn)
```

### Using Different Initializers

```python
from neural_network.core import Dense, get_initializer

# Using string names
dense1 = Dense(256, kernel_initializer='xavier_normal')
dense2 = Dense(128, kernel_initializer='he_uniform')

# Using initializer objects
from neural_network.core import HeNormal
dense3 = Dense(64, kernel_initializer=HeNormal(seed=42))
```

### Dropout and Regularization

```python
from neural_network.core import Dense, Dropout, SpatialDropout2D

# Standard dropout
dropout = Dropout(rate=0.5)

# Spatial dropout for CNNs
spatial_dropout = SpatialDropout2D(rate=0.2)

# Apply during training
x_train = dropout.forward(x, training=True)
x_test = dropout.forward(x, training=False)  # No dropout during inference
```

## Architecture

The implementation follows a modular design:

```
neural_network/
├── core/
│   ├── base.py              # Abstract base Layer class
│   ├── layers/
│   │   ├── dense.py         # Dense layer
│   │   ├── conv2d.py        # 2D Convolution
│   │   ├── pooling.py       # Pooling layers
│   │   ├── dropout.py       # Dropout variants
│   │   └── batchnorm.py     # Batch normalization
│   ├── activations/
│   │   └── activations.py   # All activation functions
│   └── initializers/
│       └── initializers.py  # Weight initializers
└── example_usage.py         # Comprehensive examples
```

## Key Implementation Details

1. **Pure NumPy**: No external dependencies beyond NumPy
2. **Modular Design**: Each component is self-contained
3. **Gradient Computation**: All layers implement proper backpropagation
4. **Memory Efficient**: Caches only necessary values for backward pass
5. **Flexible Shapes**: Supports various input dimensions
6. **CPU Optimized**: Uses NumPy's vectorized operations

## Memory Storage

Components are stored in memory at:
`swarm-auto-centralized-1750808103264/core-dev/components`

## Testing

Run the example script to verify all components:

```bash
python neural_network/example_usage.py
```

This will test:
- Dense and convolutional networks
- All activation functions
- All weight initializers
- Dropout variants
- Forward and backward propagation

## Performance Notes

- Optimized for CPU execution using NumPy's BLAS operations
- Conv2D uses im2col transformation for efficient computation
- BatchNormalization maintains moving statistics for inference
- All operations are vectorized for batch processing