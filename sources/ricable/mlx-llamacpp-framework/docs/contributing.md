# Contributing to the LLMs on Apple Silicon Project

We welcome contributions from the community! This document outlines the process for contributing to this project and guidelines to follow.

## Ways to Contribute

There are many ways to contribute to this project:

1. **Code Contributions**: Implement new features, fix bugs, or improve performance
2. **Documentation**: Improve or expand documentation, tutorials, and examples
3. **Testing**: Test the project on different hardware configurations and report results
4. **Bug Reports**: File detailed bug reports when you encounter issues
5. **Feature Requests**: Suggest new features or improvements
6. **Community Support**: Help answer questions from other users

## Development Setup

### Prerequisites

- Apple Silicon Mac (M1, M2, or M3 series)
- macOS 12 (Monterey) or newer
- Git
- Python 3.8+ (for MLX)
- C++17 compatible compiler (for llama.cpp)
- Xcode Command Line Tools

### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
   ```bash
   git clone https://github.com/YOUR-USERNAME/flow2.git
   cd flow2
   ```
3. Set up the development environment
   ```bash
   # Set up llama.cpp
   cd llama.cpp-setup
   ./scripts/setup.sh
   
   # Set up MLX
   cd ../mlx-setup
   ./scripts/setup.sh
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

## Contribution Workflow

1. **Create a branch**: Create a new branch for your contribution
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**: Implement your feature or fix the bug

3. **Follow coding standards**:
   - Use consistent code formatting
   - Add appropriate comments
   - Write tests for new functionality
   - Update documentation

4. **Test your changes**:
   - Ensure all tests pass
   - Test on different hardware configurations if possible
   - Verify backward compatibility

5. **Commit your changes**:
   ```bash
   git commit -m "Add feature: brief description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**: Go to the GitHub repository and create a pull request from your branch

## Pull Request Guidelines

When submitting a pull request, please:

- **Be clear**: Explain what your changes do and why they should be included
- **Be specific**: Link to any relevant issues
- **Be thorough**: Include screenshots, performance metrics, or other relevant information
- **Be patient**: Allow time for maintainers to review your PR

## Code Style Guidelines

- **Python Code**:
  - Follow PEP 8 style guide
  - Use type hints where appropriate
  - Document functions and classes with docstrings

- **C++ Code**:
  - Follow the project's existing style
  - Use consistent indentation (2 spaces)
  - Include comments for complex logic

- **Documentation**:
  - Use Markdown for all documentation
  - Include code examples where appropriate
  - Ensure links work correctly

## Testing Guidelines

- **Unit Tests**: Write unit tests for new functionality
- **Integration Tests**: Ensure your changes work with the rest of the system
- **Performance Tests**: For performance-critical code, include benchmarks

## Documentation Guidelines

When adding or modifying documentation:

- Use clear, concise language
- Include examples where appropriate
- Organize information logically
- Check for spelling and grammatical errors
- Ensure accuracy of technical information

## Reporting Bugs

When reporting bugs, please include:

- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Hardware configuration (Mac model, RAM, etc.)
- Software versions (macOS, Python, etc.)
- Screenshots or logs if applicable

## Feature Requests

When requesting features, please:

- Clearly describe the feature
- Explain why it would be valuable
- Provide examples of how it would be used
- Note any potential implementation challenges

## Community Guidelines

- Be respectful and inclusive
- Help others when you can
- Ask questions clearly
- Accept feedback graciously
- Give credit where it's due

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

If you have questions about contributing, please open an issue in the repository or contact the maintainers directly.

Thank you for contributing to making LLMs on Apple Silicon more accessible and powerful!