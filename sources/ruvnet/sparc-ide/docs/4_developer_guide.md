# SPARC IDE Developer Guide

This comprehensive guide provides information for developers who want to contribute to SPARC IDE, extend its functionality, or develop custom extensions.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Development Environment Setup](#development-environment-setup)
4. [Building from Source](#building-from-source)
5. [Contributing Guidelines](#contributing-guidelines)
6. [Extension Development](#extension-development)
7. [Customizing SPARC IDE](#customizing-sparc-ide)
8. [Security Considerations](#security-considerations)
9. [Testing and Quality Assurance](#testing-and-quality-assurance)
10. [Release Process](#release-process)

## Introduction

SPARC IDE is an open-source project built on VSCodium that integrates the SPARC methodology with Roo Code for AI-assisted development. This guide is intended for developers who want to:

- Contribute to the core SPARC IDE project
- Develop extensions for SPARC IDE
- Customize SPARC IDE for specific use cases
- Understand the architecture and implementation details

## Project Structure

The SPARC IDE project follows a modular structure to maintain separation of concerns and enable extensibility:

```
sparc-ide/
├── branding/           # Branding assets (icons, splash screens)
│   ├── icons/          # Application icons
│   ├── linux/          # Linux-specific branding
│   ├── macos/          # macOS-specific branding
│   ├── splash/         # Splash screen assets
│   └── windows/        # Windows-specific branding
├── docs/               # Documentation
├── extensions/         # Built-in extensions
├── package/            # Packaging configuration and artifacts
├── plans/              # Project planning documents
├── scripts/            # Build and setup scripts
├── src/                # Source code
│   ├── config/         # Configuration files
│   ├── mcp/            # MCP server implementation
│   └── sparc-workflow/ # SPARC workflow implementation
├── test-reports/       # Test reports
├── tests/              # Test scripts and utilities
└── vscodium/           # VSCodium source code (cloned during setup)
```

### Key Components

- **VSCodium Base**: The foundation of SPARC IDE, providing the core editor functionality
- **Roo Code Integration**: Integration with Roo Code for AI-assisted development
- **SPARC Workflow**: Implementation of the SPARC methodology
- **AI Integration**: Support for multiple AI models and custom modes
- **UI/UX Customizations**: Custom themes, layouts, and keybindings
- **MCP Server**: Model Context Protocol server for enhanced AI capabilities

## Development Environment Setup

### Prerequisites

Before setting up the development environment, ensure you have the following tools installed:

#### All Platforms
- **Git**: For cloning the repository
- **Node.js**: Version 18 or later
- **Yarn**: Version 1.22 or later

#### Linux-specific
Install the following packages:

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install build-essential libx11-dev libxkbfile-dev libsecret-1-dev fakeroot rpm
```

**Fedora**:
```bash
sudo dnf install make gcc gcc-c++ libX11-devel libxkbfile-devel libsecret-devel rpm-build
```

#### Windows-specific
- **Visual Studio Build Tools**: For compiling native modules
  - Install from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Required components: "Desktop development with C++"

#### macOS-specific
- **Xcode Command Line Tools**: For compiling native modules
  ```bash
  xcode-select --install
  ```

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

This setup script performs the following tasks:
- Sets up the VSCodium build environment
- Applies SPARC IDE branding
- Downloads the Roo Code extension
- Configures the UI
- Sets up the MCP server
- Creates SPARC workflow directories

## Building from Source

### Building SPARC IDE

After setting up the development environment, you can build SPARC IDE:

```bash
chmod +x scripts/build-sparc-ide.sh
./scripts/build-sparc-ide.sh
```

This script will:
1. Build VSCodium with SPARC IDE customizations
2. Apply branding assets
3. Install built-in extensions
4. Configure default settings
5. Create the final application package

The build artifacts will be available in the `dist/` directory.

### Running the Development Version

To run the development version of SPARC IDE:

```bash
./dist/sparc-ide
```

### Building for Different Platforms

To build SPARC IDE for a specific platform:

```bash
./scripts/build-sparc-ide.sh --platform [linux|windows|macos]
```

### Creating Installation Packages

To create installation packages for distribution:

```bash
chmod +x scripts/package-sparc-ide.sh
./scripts/package-sparc-ide.sh
```

The packages will be available in the `package/` directory, organized by platform.

## Contributing Guidelines

### Development Workflow

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

3. Make your changes, following our coding standards

4. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request from your fork to the main repository

### Pull Request Process

1. Ensure your PR includes a clear description of the changes and the purpose
2. Update documentation as needed
3. Include tests for new features or bug fixes
4. Ensure all tests pass
5. Make sure your code follows our coding standards
6. Link any related issues in the PR description
7. Request a review from maintainers

### Coding Standards

#### General Guidelines

- Write clean, readable, and maintainable code
- Follow the principle of "Do One Thing" (DOT) for functions and classes
- Keep functions small and focused
- Use meaningful variable and function names
- Comment your code when necessary, especially for complex logic
- Write self-documenting code where possible

#### JavaScript/TypeScript

- Follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use ES6+ features where appropriate
- Use TypeScript for type safety
- Use async/await for asynchronous code
- Prefer const over let, and avoid var
- Use destructuring where it improves readability

#### CSS/SCSS

- Follow the [BEM methodology](http://getbem.com/) for CSS class naming
- Use SCSS for styling
- Keep selectors simple and avoid deep nesting
- Use variables for colors, fonts, and other repeated values

#### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Extension Development

SPARC IDE supports extensions using the VS Code extension API. This section explains how to develop extensions specifically for SPARC IDE.

### Extension Types

You can develop several types of extensions for SPARC IDE:

1. **UI Extensions**: Customize the user interface
2. **Language Extensions**: Add support for new programming languages
3. **Tool Extensions**: Integrate with external tools
4. **AI Extensions**: Enhance AI capabilities
5. **SPARC Extensions**: Extend the SPARC methodology

### Creating a New Extension

To create a new extension for SPARC IDE:

1. Install the VS Code Extension Generator:
   ```bash
   npm install -g yo generator-code
   ```

2. Generate a new extension:
   ```bash
   yo code
   ```

3. Follow the prompts to configure your extension

4. Develop your extension using the VS Code Extension API

### SPARC IDE-Specific APIs

SPARC IDE provides additional APIs for extensions to interact with SPARC-specific features:

#### SPARC Workflow API

```typescript
// Access the SPARC Workflow API
const sparcWorkflow = vscode.extensions.getExtension('sparc-ide.sparc-workflow').exports;

// Get the current phase
const currentPhase = await sparcWorkflow.getCurrentPhase();

// Switch to a different phase
await sparcWorkflow.switchPhase('architecture');

// Create an artifact
await sparcWorkflow.createArtifact('specification', 'path/to/artifact.md', 'requirements');
```

#### AI Integration API

```typescript
// Access the AI Integration API
const aiIntegration = vscode.extensions.getExtension('sparc-ide.ai-integration').exports;

// Get the current AI model
const currentModel = await aiIntegration.getCurrentModel();

// Switch to a different AI model
await aiIntegration.switchModel('claude');

// Execute a prompt
const result = await aiIntegration.executePrompt('Generate a function to sort an array', {
  temperature: 0.7,
  maxTokens: 1000
});
```

### Testing Extensions

To test your extension with SPARC IDE:

1. Package your extension:
   ```bash
   vsce package
   ```

2. Install the extension in SPARC IDE:
   ```bash
   ./dist/sparc-ide --install-extension your-extension.vsix
   ```

3. Test your extension functionality

### Publishing Extensions

Extensions for SPARC IDE can be published to the VS Code Marketplace or distributed as VSIX files:

1. Create a publisher account on the [VS Code Marketplace](https://marketplace.visualstudio.com/VSCode)
2. Publish your extension:
   ```bash
   vsce publish
   ```

## Customizing SPARC IDE

SPARC IDE can be customized in various ways to meet specific requirements:

### Customizing Branding

To customize the branding of SPARC IDE:

1. Modify the assets in the `branding/` directory:
   - Replace icons in `branding/icons/`
   - Update splash screens in `branding/splash/`
   - Modify platform-specific assets in `branding/linux/`, `branding/macos/`, and `branding/windows/`

2. Update product information in `src/config/product.json`:
   - Change the product name, description, and other metadata
   - Update URLs for documentation, support, etc.

3. Rebuild SPARC IDE with the new branding:
   ```bash
   ./scripts/apply-branding.sh
   ./scripts/build-sparc-ide.sh
   ```

### Customizing Default Settings

To customize the default settings of SPARC IDE:

1. Modify `src/config/settings.json`:
   - Update editor settings
   - Configure UI preferences
   - Set default extensions
   - Configure AI integration

2. Rebuild SPARC IDE with the new settings:
   ```bash
   ./scripts/build-sparc-ide.sh
   ```

### Customizing Keybindings

To customize the default keybindings of SPARC IDE:

1. Modify `src/config/keybindings.json`:
   - Update existing keybindings
   - Add new keybindings
   - Remove unwanted keybindings

2. Rebuild SPARC IDE with the new keybindings:
   ```bash
   ./scripts/build-sparc-ide.sh
   ```

### Customizing SPARC Workflow

To customize the SPARC workflow:

1. Modify `src/config/product.json` to update the SPARC phases:
   ```json
   "sparcConfig": {
     "phases": [
       {
         "id": "specification",
         "name": "Specification",
         "description": "Define detailed requirements and acceptance criteria",
         "templates": ["requirements.md", "user-stories.md", "acceptance-criteria.md"],
         "aiPrompts": ["Generate requirements", "Create user stories", "Define acceptance criteria"]
       },
       // Add or modify phases here
     ]
   }
   ```

2. Update templates in `src/sparc-workflow/templates/`:
   - Modify existing templates
   - Add new templates
   - Update template content

3. Rebuild SPARC IDE with the new workflow:
   ```bash
   ./scripts/build-sparc-ide.sh
   ```

## Security Considerations

### Secure Development Practices

When developing for SPARC IDE, follow these security best practices:

1. **Input Validation**: Validate all user inputs to prevent injection attacks
2. **Secure API Usage**: Use secure methods for API calls and data handling
3. **Dependency Management**: Keep dependencies updated and use vulnerability scanning
4. **Secure Storage**: Use secure storage for sensitive information
5. **Principle of Least Privilege**: Request only the permissions your extension needs

### API Key Management

SPARC IDE uses API keys for AI services. When developing features that use these services:

1. Never hardcode API keys in your code
2. Use the secure storage API for storing API keys:
   ```typescript
   // Store an API key securely
   await vscode.secrets.store('your-extension.apiKey', 'your-api-key');
   
   // Retrieve an API key
   const apiKey = await vscode.secrets.get('your-extension.apiKey');
   ```

3. Use environment variables for API keys during development:
   ```typescript
   const apiKey = process.env.YOUR_API_KEY || await vscode.secrets.get('your-extension.apiKey');
   ```

### Extension Verification

SPARC IDE includes extension verification to ensure the integrity and authenticity of extensions:

1. Sign your extensions with a code signing certificate
2. Provide a verification record with your extension
3. Follow the verification protocol documented in `docs/security-enhancements.md`

## Testing and Quality Assurance

### Testing Frameworks

SPARC IDE uses the following testing frameworks:

- **Jest**: For unit and integration tests
- **Mocha**: For additional test scenarios
- **Playwright**: For end-to-end tests

### Running Tests

To run tests for SPARC IDE:

```bash
# Run all tests
./tests/run_all_tests.sh

# Run specific test categories
./tests/run-tests.sh branding
./tests/run-tests.sh build-scripts
./tests/run-tests.sh roo-code
./tests/run-tests.sh ui-config
```

### Writing Tests

When writing tests for SPARC IDE:

1. **Unit Tests**: Test individual functions and components
   ```typescript
   // Example unit test
   describe('SPARC Phase Management', () => {
     it('should switch to the specified phase', async () => {
       const result = await phaseManager.switchPhase('architecture');
       expect(result.id).toBe('architecture');
     });
   });
   ```

2. **Integration Tests**: Test interactions between components
   ```typescript
   // Example integration test
   describe('AI Integration with SPARC Workflow', () => {
     it('should generate content based on the current phase', async () => {
       await sparcWorkflow.switchPhase('specification');
       const result = await aiIntegration.executePrompt('Generate requirements');
       expect(result.text).toContain('Requirements');
     });
   });
   ```

3. **End-to-End Tests**: Test the complete user experience
   ```typescript
   // Example end-to-end test
   test('should create a new project with SPARC workflow', async ({ page }) => {
     await page.click('button:has-text("New Project")');
     await page.fill('input[name="projectName"]', 'Test Project');
     await page.click('button:has-text("Create")');
     await expect(page.locator('.sparc-phase-indicator')).toHaveText('Specification');
   });
   ```

### Continuous Integration

SPARC IDE uses GitHub Actions for continuous integration:

1. **Build Workflow**: Builds SPARC IDE for all platforms
2. **Test Workflow**: Runs all tests
3. **Package Workflow**: Creates installation packages
4. **Release Workflow**: Publishes releases

The CI configuration is defined in `.github/workflows/`.

## Release Process

### Version Management

SPARC IDE follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes

Version information is stored in `package.json` and `src/config/product.json`.

### Release Preparation

To prepare a release:

1. Update version numbers:
   ```bash
   # Update version in package.json and product.json
   ./scripts/update-version.sh 1.2.3
   ```

2. Update the changelog:
   - Add a new section for the release
   - Document new features, bug fixes, and other changes
   - Include contributor acknowledgments

3. Create a release branch:
   ```bash
   git checkout -b release/1.2.3
   git add .
   git commit -m "Prepare release 1.2.3"
   git push origin release/1.2.3
   ```

### Release Build

To build a release:

1. Merge the release branch to main:
   ```bash
   git checkout main
   git merge release/1.2.3
   git push origin main
   ```

2. Tag the release:
   ```bash
   git tag -a v1.2.3 -m "Release 1.2.3"
   git push origin v1.2.3
   ```

3. The release workflow will automatically:
   - Build SPARC IDE for all platforms
   - Create installation packages
   - Generate release notes
   - Publish the release on GitHub

### Post-Release

After a release:

1. Update the documentation website
2. Announce the release on social media and community channels
3. Start planning for the next release