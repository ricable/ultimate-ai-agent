# Contributing to SPARC IDE

Thank you for your interest in contributing to SPARC IDE! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Pull Request Process](#pull-request-process)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Issue Reporting](#issue-reporting)
9. [Feature Requests](#feature-requests)
10. [Community](#community)

## Code of Conduct

SPARC IDE is committed to fostering an open and welcoming environment. By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

Before you begin, ensure you have met the requirements listed in the [Installation Guide](installation-guide.md#prerequisites).

### Setting Up the Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/sparc-ide.git
   cd sparc-ide
   ```
3. Add the original repository as a remote:
   ```bash
   git remote add upstream https://github.com/sparc-ide/sparc-ide.git
   ```
4. Set up the development environment:
   ```bash
   chmod +x scripts/setup-sparc-ide.sh
   ./scripts/setup-sparc-ide.sh
   ```

## Development Workflow

We follow a feature branch workflow:

1. Ensure your fork is up to date:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

3. Make your changes, following our [coding standards](#coding-standards)

4. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request from your fork to the main repository

## Pull Request Process

1. Ensure your PR includes a clear description of the changes and the purpose
2. Update documentation as needed
3. Include tests for new features or bug fixes
4. Ensure all tests pass
5. Make sure your code follows our coding standards
6. Link any related issues in the PR description
7. Request a review from maintainers

Pull requests will be reviewed by maintainers, who may request changes or provide feedback. Once approved, your PR will be merged into the main branch.

## Coding Standards

### General Guidelines

- Write clean, readable, and maintainable code
- Follow the principle of "Do One Thing" (DOT) for functions and classes
- Keep functions small and focused
- Use meaningful variable and function names
- Comment your code when necessary, especially for complex logic
- Write self-documenting code where possible

### JavaScript/TypeScript

- Follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use ES6+ features where appropriate
- Use TypeScript for type safety
- Use async/await for asynchronous code
- Prefer const over let, and avoid var
- Use destructuring where it improves readability

### CSS/SCSS

- Follow the [BEM methodology](http://getbem.com/) for CSS class naming
- Use SCSS for styling
- Keep selectors simple and avoid deep nesting
- Use variables for colors, fonts, and other repeated values

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Testing

We use the following testing frameworks:

- Jest for unit and integration tests
- Playwright for end-to-end tests

### Running Tests

```bash
# Run all tests
npm test

# Run unit tests
npm run test:unit

# Run integration tests
npm run test:integration

# Run end-to-end tests
npm run test:e2e
```

### Writing Tests

- Write tests for all new features and bug fixes
- Follow the AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Keep tests focused and small
- Use descriptive test names that explain what is being tested

## Documentation

Good documentation is crucial for the project. Please update documentation when:

- Adding new features
- Changing existing functionality
- Fixing bugs that might affect user behavior
- Adding or changing API endpoints

### Documentation Guidelines

- Use clear, concise language
- Include examples where appropriate
- Use proper Markdown formatting
- Keep documentation up to date with code changes
- Document both user-facing features and internal APIs

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Screenshots if applicable
6. Your environment (OS, browser, SPARC IDE version)
7. Any additional context

### Security Issues

If you discover a security vulnerability, please do NOT open an issue. Email [security@sparc-ide.org](mailto:security@sparc-ide.org) instead.

## Feature Requests

Feature requests are welcome! When submitting a feature request:

1. Use a clear, descriptive title
2. Provide a detailed description of the feature
3. Explain why this feature would be useful
4. Provide examples of how the feature would work
5. Consider including mockups or diagrams

## Community

Join our community to get help, share ideas, and collaborate:

- [Discord](https://discord.gg/sparc-ide)
- [GitHub Discussions](https://github.com/sparc-ide/sparc-ide/discussions)
- [Twitter](https://twitter.com/sparc_ide)

## Project Structure

Understanding the project structure will help you contribute effectively:

```
sparc-ide/
├── branding/           # Branding assets
├── docs/               # Documentation
├── extensions/         # Built-in extensions
├── scripts/            # Build and setup scripts
├── src/
│   ├── config/         # Configuration files
│   ├── mcp/            # MCP server
│   └── sparc-workflow/ # SPARC workflow implementation
└── vscodium/           # VSCodium source code (cloned during setup)
```

## Development Tips

### Building and Running

To build and run SPARC IDE during development:

```bash
# Build SPARC IDE
./scripts/build-sparc-ide.sh

# Run SPARC IDE
./dist/sparc-ide
```

### Debugging

You can debug SPARC IDE by running it with the `--inspect` flag:

```bash
./dist/sparc-ide --inspect
```

Then connect to the debugger using Chrome DevTools or VS Code.

### Working with VSCodium

Since SPARC IDE is built on VSCodium, you may need to make changes to the VSCodium source code. The setup script clones VSCodium into the `vscodium` directory. See the [VSCodium documentation](https://github.com/VSCodium/vscodium/wiki) for more information.

## Acknowledgements

Thank you to all contributors who help make SPARC IDE better! Your contributions, whether code, documentation, bug reports, or feature requests, are greatly appreciated.