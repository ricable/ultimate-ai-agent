# Neuro-Divergent User Guide

Welcome to the comprehensive user guide for Neuro-Divergent, a high-performance neural forecasting library built in Rust that provides 100% compatibility with Python's NeuralForecast library.

## What is Neuro-Divergent?

Neuro-Divergent combines the performance and safety benefits of Rust with the familiar API of Python's NeuralForecast. Built on the ruv-FANN neural network foundation, it offers state-of-the-art neural forecasting capabilities with zero-cost abstractions and compile-time guarantees.

## Key Features

- **100% NeuralForecast API Compatibility**: Drop-in replacement for Python users
- **High Performance**: Rust performance with SIMD optimization  
- **Memory Safety**: Zero-cost abstractions with compile-time guarantees
- **Async Support**: Asynchronous training and prediction
- **27+ Model Support**: From simple MLPs to advanced Transformers
- **Extensible Architecture**: Easy to add custom models and components

## Documentation Structure

This user guide is organized into progressive sections to help users of all skill levels:

### Getting Started
- [Installation Guide](installation.md) - Set up Neuro-Divergent in your environment
- [Quick Start](quick-start.md) - Your first forecast in under 5 minutes
- [Basic Concepts](basic-concepts.md) - Core concepts and terminology

### Model Documentation
- [Model Overview](models/index.md) - All 27+ available models
- [Basic Models](models/basic-models.md) - MLP, DLinear, NLinear
- [Recurrent Models](models/recurrent-models.md) - RNN, LSTM, GRU
- [Advanced Models](models/advanced-models.md) - NBEATS, NHiTS, NBEATSx
- [Transformer Models](models/transformer-models.md) - TFT, Informer, Autoformer
- [Specialized Models](models/specialized-models.md) - DeepAR, TCN, TimesNet

### Core Functionality
- [Data Handling](data-handling.md) - Data formats, preprocessing, feature engineering
- [Training](training.md) - Training workflow, hyperparameter tuning, optimization
- [Prediction](prediction.md) - Single/batch prediction, probabilistic forecasting
- [Evaluation](evaluation.md) - Metrics, validation strategies, performance analysis

### Advanced Usage
- [Best Practices](best-practices.md) - Model selection, optimization, deployment
- [Performance](performance.md) - Memory optimization, scaling, acceleration
- [Advanced Usage](advanced-usage.md) - Custom models, async operations, integrations
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [FAQ](faq.md) - Frequently asked questions

## Target Audiences

This documentation is designed for multiple audiences:

### Beginners
New to time series forecasting who want to learn the fundamentals while using a production-ready library.

### Data Scientists  
Familiar with machine learning but new to Rust who want high-performance forecasting capabilities.

### Rust Developers
Experienced with Rust but new to time series forecasting who want to leverage Rust's performance benefits.

### Experts
Advanced users looking for optimization tips, custom implementations, and production deployment strategies.

## Quick Navigation

### I want to...
- **Get started quickly**: Go to [Quick Start](quick-start.md)
- **Install the library**: See [Installation](installation.md)  
- **Choose a model**: Check [Model Overview](models/index.md)
- **Load my data**: Read [Data Handling](data-handling.md)
- **Train a model**: Follow [Training](training.md)
- **Make predictions**: See [Prediction](prediction.md)
- **Optimize performance**: Read [Performance](performance.md)
- **Deploy to production**: Check [Best Practices](best-practices.md)

### I'm coming from...
- **Python NeuralForecast**: The API is identical - start with [Quick Start](quick-start.md)
- **Other Rust ML libraries**: See [Basic Concepts](basic-concepts.md) for our approach
- **Traditional forecasting**: Check [Model Overview](models/index.md) to understand neural approaches
- **Deep learning frameworks**: See [Advanced Usage](advanced-usage.md) for custom implementations

## Community and Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions for help and sharing
- **Documentation**: This guide is continuously updated with user feedback
- **Examples**: Check the `examples/` directory for practical use cases

## What's Next?

Ready to start forecasting? Choose your path:

1. **Complete Beginner**: Start with [Installation](installation.md) → [Basic Concepts](basic-concepts.md) → [Quick Start](quick-start.md)

2. **NeuralForecast User**: Jump to [Quick Start](quick-start.md) to see the familiar API in Rust

3. **Rust Developer**: Begin with [Basic Concepts](basic-concepts.md) to understand forecasting fundamentals

4. **Looking for a specific model**: Go directly to [Model Overview](models/index.md)

Let's build the future of neural forecasting together!