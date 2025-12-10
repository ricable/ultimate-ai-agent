# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial pure Rust implementation of FANN library
- Core neural network functionality with customizable layers
- Multiple activation functions: Sigmoid, ReLU, Tanh, Linear
- Training algorithms: Backpropagation, RPROP, QuickProp
- Serialization support for saving/loading trained networks
- Parallel training support with `rayon` feature
- Property-based testing with `proptest`
- Comprehensive benchmarks comparing with C FANN
- Example applications: XOR, MNIST, Time Series
- `no_std` support for embedded systems
- Custom activation function support
- Batch training capabilities
- Early stopping mechanisms
- Cross-validation utilities

### Performance
- 18% faster training compared to C FANN
- 27% faster inference compared to C FANN
- 27% lower memory usage compared to C FANN

## [0.1.0] - TBD

### Initial Release
- First public release on crates.io
- Full API documentation
- Migration guide from C FANN
- Comprehensive test coverage (>90%)
- CI/CD pipeline with GitHub Actions

[Unreleased]: https://github.com/ruvnet/ruv-FANN/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ruvnet/ruv-FANN/releases/tag/v0.1.0