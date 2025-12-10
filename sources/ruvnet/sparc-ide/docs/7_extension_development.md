# SPARC IDE Extension Development Guide

This comprehensive guide explains how to develop custom extensions for SPARC IDE, integrate with existing features, and publish your extensions for others to use.

## Table of Contents

1. [Introduction to SPARC IDE Extensions](#introduction-to-sparc-ide-extensions)
2. [Setting Up the Development Environment](#setting-up-the-development-environment)
3. [Creating Your First Extension](#creating-your-first-extension)
4. [Extension API Overview](#extension-api-overview)
5. [Integrating with SPARC Workflow](#integrating-with-sparc-workflow)
6. [Integrating with Roo Code](#integrating-with-roo-code)
7. [UI Extensions](#ui-extensions)
8. [AI-Powered Extensions](#ai-powered-extensions)
9. [Testing and Debugging](#testing-and-debugging)
10. [Publishing Your Extension](#publishing-your-extension)

## Introduction to SPARC IDE Extensions

SPARC IDE is built on VSCodium, which provides a powerful extension system that allows developers to add new features, integrate with external tools, and customize the user experience. This guide will help you create extensions specifically for SPARC IDE, leveraging its unique features like the SPARC workflow and Roo Code integration.

### Types of Extensions

You can develop several types of extensions for SPARC IDE:

1. **SPARC Workflow Extensions**: Extend the SPARC methodology with custom phases, templates, or tools
2. **Roo Code Extensions**: Enhance the AI capabilities with custom prompts, tools, or integrations
3. **UI Extensions**: Customize the user interface with new views, panels, or themes
4. **Language Extensions**: Add support for new programming languages or frameworks
5. **Tool Extensions**: Integrate with external tools and services
6. **AI-Powered Extensions**: Create extensions that leverage AI capabilities

### Extension Architecture

SPARC IDE extensions follow the same architecture as VS Code extensions:

```
Extension
├── package.json       # Extension manifest
├── README.md          # Documentation
├── CHANGELOG.md       # Version history
├── src/               # Source code
│   ├── extension.ts   # Extension entry point
│   └── ...            # Other source files
├── dist/              # Compiled code (for TypeScript)
├── node_modules/      # Dependencies
└── .vscodeignore      # Files to exclude from package
```

## Setting Up the Development Environment

Before you start developing extensions for SPARC IDE, you need to set up your development environment:

### Prerequisites

- **Node.js**: Version 18 or later
- **npm**: Version 8 or later
- **Git**: For version control
- **SPARC IDE**: For testing your extension
- **VS Code Extension Generator**: For scaffolding your extension

### Installing the VS Code Extension Generator

Install the VS Code Extension Generator using npm:

```bash
npm install -g yo generator-code
```

### Setting Up Your Development Workspace

1. Create a new directory for your extension:
   ```bash
   mkdir my-sparc-extension
   cd my-sparc-extension
   ```

2. Initialize a new extension project:
   ```bash
   yo code
   ```

3. Follow the prompts to configure your extension:
   - Select "New Extension (TypeScript)" for TypeScript or "New Extension (JavaScript)" for JavaScript
   - Enter a name for your extension
   - Enter a description
   - Enter your publisher name (create one on the VS Code Marketplace if you don't have one)
   - Choose whether to initialize a Git repository

4. Open the project in SPARC IDE:
   ```bash
   code .
   ```

## Creating Your First Extension

Let's create a simple "Hello SPARC" extension to understand the basics:

### Extension Manifest

The `package.json` file is the manifest for your extension. It defines metadata, dependencies, activation events, and contribution points:

```json
{
  "name": "hello-sparc",
  "displayName": "Hello SPARC",
  "description": "A simple extension for SPARC IDE",
  "version": "0.0.1",
  "engines": {
    "vscode": "^1.60.0"
  },
  "categories": [
    "Other"
  ],
  "activationEvents": [
    "onCommand:hello-sparc.helloWorld"
  ],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "hello-sparc.helloWorld",
        "title": "Hello SPARC: Say Hello"
      }
    ]
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./",
    "pretest": "npm run compile && npm run lint",
    "lint": "eslint src --ext ts",
    "test": "node ./out/test/runTest.js"
  },
  "devDependencies": {
    "@types/vscode": "^1.60.0",
    "@types/glob": "^7.1.3",
    "@types/mocha": "^8.2.2",
    "@types/node": "14.x",
    "eslint": "^7.27.0",
    "glob": "^7.1.7",
    "mocha": "^8.4.0",
    "typescript": "^4.3.2",
    "vscode-test": "^1.5.2"
  }
}
```

### Extension Entry Point

The entry point for your extension is typically `src/extension.ts` (for TypeScript) or `src/extension.js` (for JavaScript):

```typescript
// src/extension.ts
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  console.log('Congratulations, your extension "hello-sparc" is now active!');

  let disposable = vscode.commands.registerCommand('hello-sparc.helloWorld', () => {
    vscode.window.showInformationMessage('Hello from SPARC IDE!');
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}
```

### Building and Running Your Extension

1. Compile your extension:
   ```bash
   npm run compile
   ```

2. Press F5 to launch a new SPARC IDE instance with your extension loaded

3. Open the Command Palette (Ctrl+Shift+P) and run "Hello SPARC: Say Hello"

4. You should see a notification saying "Hello from SPARC IDE!"

## Extension API Overview

SPARC IDE extends the VS Code API with additional APIs for SPARC-specific features:

### VS Code API

The core VS Code API provides access to:

- **Editor**: Manipulate text documents and editors
- **Workspace**: Access workspace folders and files
- **Languages**: Work with language features like IntelliSense
- **Debug**: Integrate with the debugging system
- **UI**: Create custom UI components
- **Extensions**: Interact with other extensions

### SPARC-Specific APIs

SPARC IDE adds the following APIs:

- **SPARC Workflow API**: Interact with the SPARC methodology
- **Roo Code API**: Integrate with AI capabilities
- **AI Integration API**: Work with multiple AI models
- **MCP Server API**: Access the Model Context Protocol server

### Accessing SPARC APIs

To access SPARC-specific APIs, you need to get the API from the corresponding extension:

```typescript
// Access the SPARC Workflow API
const sparcWorkflow = vscode.extensions.getExtension('sparc-ide.sparc-workflow')?.exports;

// Access the Roo Code API
const rooCode = vscode.extensions.getExtension('sparc-ide.roo-code')?.exports;

// Access the AI Integration API
const aiIntegration = vscode.extensions.getExtension('sparc-ide.ai-integration')?.exports;
```

## Integrating with SPARC Workflow

The SPARC Workflow API allows you to interact with the SPARC methodology:

### SPARC Workflow API Overview

```typescript
interface SPARCWorkflowAPI {
  // Phase management
  getCurrentPhase(): Promise<SPARCPhase>;
  switchPhase(phaseId: string): Promise<SPARCPhase>;
  getPhases(): Promise<SPARCPhase[]>;
  
  // Template management
  getTemplates(phaseId: string): Promise<Template[]>;
  createFromTemplate(templateId: string, path: string): Promise<vscode.Uri>;
  
  // Artifact management
  createArtifact(phaseId: string, path: string, type: string): Promise<Artifact>;
  getArtifacts(phaseId: string): Promise<Artifact[]>;
  
  // Progress monitoring
  getProgress(): Promise<WorkflowProgress>;
  updateProgress(phaseId: string, progress: number): Promise<WorkflowProgress>;
}
```

### Example: Creating a Custom SPARC Phase

Here's an example of how to create an extension that adds a custom phase to the SPARC workflow:

```typescript
import * as vscode from 'vscode';

export async function activate(context: vscode.ExtensionContext) {
  // Access the SPARC Workflow API
  const sparcWorkflow = vscode.extensions.getExtension('sparc-ide.sparc-workflow')?.exports;
  
  if (!sparcWorkflow) {
    vscode.window.showErrorMessage('SPARC Workflow API not available');
    return;
  }
  
  // Register a custom phase
  const customPhase = {
    id: 'validation',
    name: 'Validation',
    description: 'Validate the implementation against requirements',
    templates: ['validation-plan.md', 'validation-report.md'],
    aiPrompts: ['Generate validation plan', 'Create validation tests'],
    status: 'not-started',
    artifacts: [],
    nextPhase: 'completion',
    previousPhase: 'refinement'
  };
  
  // Register the custom phase
  await sparcWorkflow.registerPhase(customPhase);
  
  // Register templates for the custom phase
  const validationPlanTemplate = {
    id: 'validation-plan',
    name: 'Validation Plan',
    description: 'Plan for validating the implementation',
    content: `# Validation Plan\n\n## Objectives\n\n## Test Cases\n\n## Validation Criteria\n\n## Timeline\n`,
    phaseId: 'validation'
  };
  
  const validationReportTemplate = {
    id: 'validation-report',
    name: 'Validation Report',
    description: 'Report of validation results',
    content: `# Validation Report\n\n## Summary\n\n## Test Results\n\n## Issues Found\n\n## Recommendations\n`,
    phaseId: 'validation'
  };
  
  await sparcWorkflow.registerTemplate(validationPlanTemplate);
  await sparcWorkflow.registerTemplate(validationReportTemplate);
  
  // Register a command to switch to the custom phase
  let disposable = vscode.commands.registerCommand('custom-sparc-phase.switchToValidation', async () => {
    await sparcWorkflow.switchPhase('validation');
    vscode.window.showInformationMessage('Switched to Validation phase');
  });
  
  context.subscriptions.push(disposable);
}
```

## Integrating with Roo Code

The Roo Code API allows you to integrate with AI capabilities:

### Roo Code API Overview

```typescript
interface RooCodeAPI {
  // Chat interface
  openChat(): Promise<void>;
  sendMessage(message: string): Promise<ChatResponse>;
  
  // Code generation
  generateCode(prompt: string, options?: CodeGenerationOptions): Promise<CodeGenerationResult>;
  
  // Code explanation
  explainCode(code: string, options?: ExplanationOptions): Promise<ExplanationResult>;
  
  // Code refactoring
  refactorCode(code: string, instructions: string, options?: RefactoringOptions): Promise<RefactoringResult>;
  
  // Custom prompts
  registerCustomPrompt(id: string, prompt: string, description: string): Promise<void>;
  executeCustomPrompt(id: string, variables?: Record<string, string>): Promise<PromptResult>;
}
```

### Example: Creating a Custom Roo Code Tool

Here's an example of how to create an extension that adds a custom tool to Roo Code:

```typescript
import * as vscode from 'vscode';

export async function activate(context: vscode.ExtensionContext) {
  // Access the Roo Code API
  const rooCode = vscode.extensions.getExtension('sparc-ide.roo-code')?.exports;
  
  if (!rooCode) {
    vscode.window.showErrorMessage('Roo Code API not available');
    return;
  }
  
  // Register a custom prompt for generating unit tests
  await rooCode.registerCustomPrompt(
    'generate-unit-tests',
    `Generate comprehensive unit tests for the following code. 
     Use the appropriate testing framework based on the language.
     Include tests for edge cases and error conditions.
     Code: {{code}}`,
    'Generate unit tests for selected code'
  );
  
  // Register a command to execute the custom prompt
  let disposable = vscode.commands.registerCommand('custom-roo-tool.generateTests', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('No active editor');
      return;
    }
    
    const selection = editor.selection;
    const code = editor.document.getText(selection);
    
    if (!code) {
      vscode.window.showErrorMessage('No code selected');
      return;
    }
    
    // Execute the custom prompt
    const result = await rooCode.executeCustomPrompt('generate-unit-tests', { code });
    
    // Create a new file for the tests
    const testFilePath = editor.document.uri.path.replace(/\.\w+$/, '.test$&');
    const testFileUri = vscode.Uri.file(testFilePath);
    
    const edit = new vscode.WorkspaceEdit();
    edit.createFile(testFileUri, { overwrite: true });
    edit.insert(testFileUri, new vscode.Position(0, 0), result.text);
    
    await vscode.workspace.applyEdit(edit);
    await vscode.window.showTextDocument(testFileUri);
    
    vscode.window.showInformationMessage('Unit tests generated');
  });
  
  context.subscriptions.push(disposable);
}
```

## UI Extensions

SPARC IDE allows you to create custom UI components to enhance the user experience:

### WebView API

The WebView API allows you to create custom views with HTML, CSS, and JavaScript:

```typescript
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  // Register a command to open the custom view
  let disposable = vscode.commands.registerCommand('custom-ui.openDashboard', () => {
    // Create and show a new webview panel
    const panel = vscode.window.createWebviewPanel(
      'sparcDashboard',
      'SPARC Dashboard',
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')]
      }
    );
    
    // Set the HTML content
    panel.webview.html = getWebviewContent(context, panel.webview);
    
    // Handle messages from the webview
    panel.webview.onDidReceiveMessage(
      message => {
        switch (message.command) {
          case 'alert':
            vscode.window.showInformationMessage(message.text);
            return;
        }
      },
      undefined,
      context.subscriptions
    );
  });
  
  context.subscriptions.push(disposable);
}

function getWebviewContent(context: vscode.ExtensionContext, webview: vscode.Webview): string {
  // Create a URI to a script file in the extension
  const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'media', 'main.js'));
  const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'media', 'style.css'));
  
  return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>SPARC Dashboard</title>
      <link href="${styleUri}" rel="stylesheet">
    </head>
    <body>
      <h1>SPARC Dashboard</h1>
      <div id="dashboard-content">
        <div class="card">
          <h2>Current Phase</h2>
          <div id="current-phase">Loading...</div>
        </div>
        <div class="card">
          <h2>Progress</h2>
          <div id="progress">Loading...</div>
        </div>
        <div class="card">
          <h2>Recent Artifacts</h2>
          <div id="artifacts">Loading...</div>
        </div>
      </div>
      <script src="${scriptUri}"></script>
    </body>
    </html>`;
}
```

### Tree View API

The Tree View API allows you to create custom tree views in the sidebar:

```typescript
import * as vscode from 'vscode';

class SPARCArtifactsProvider implements vscode.TreeDataProvider<ArtifactItem> {
  private _onDidChangeTreeData: vscode.EventEmitter<ArtifactItem | undefined | null | void> = new vscode.EventEmitter<ArtifactItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<ArtifactItem | undefined | null | void> = this._onDidChangeTreeData.event;
  
  constructor(private sparcWorkflow: any) {}
  
  refresh(): void {
    this._onDidChangeTreeData.fire();
  }
  
  getTreeItem(element: ArtifactItem): vscode.TreeItem {
    return element;
  }
  
  async getChildren(element?: ArtifactItem): Promise<ArtifactItem[]> {
    if (!element) {
      // Root level: show phases
      const phases = await this.sparcWorkflow.getPhases();
      return phases.map(phase => new ArtifactItem(
        phase.name,
        phase.id,
        vscode.TreeItemCollapsibleState.Collapsed,
        {
          command: 'sparc-artifacts.selectPhase',
          title: 'Select Phase',
          arguments: [phase.id]
        }
      ));
    } else {
      // Phase level: show artifacts
      const artifacts = await this.sparcWorkflow.getArtifacts(element.id);
      return artifacts.map(artifact => new ArtifactItem(
        artifact.path.split('/').pop() || '',
        artifact.id,
        vscode.TreeItemCollapsibleState.None,
        {
          command: 'sparc-artifacts.openArtifact',
          title: 'Open Artifact',
          arguments: [artifact.path]
        }
      ));
    }
  }
}

class ArtifactItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly id: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly command?: vscode.Command
  ) {
    super(label, collapsibleState);
  }
}

export async function activate(context: vscode.ExtensionContext) {
  // Access the SPARC Workflow API
  const sparcWorkflow = vscode.extensions.getExtension('sparc-ide.sparc-workflow')?.exports;
  
  if (!sparcWorkflow) {
    vscode.window.showErrorMessage('SPARC Workflow API not available');
    return;
  }
  
  // Create the tree data provider
  const artifactsProvider = new SPARCArtifactsProvider(sparcWorkflow);
  
  // Register the tree view
  const treeView = vscode.window.createTreeView('sparcArtifacts', {
    treeDataProvider: artifactsProvider
  });
  
  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('sparc-artifacts.refresh', () => artifactsProvider.refresh()),
    vscode.commands.registerCommand('sparc-artifacts.selectPhase', (phaseId) => {
      sparcWorkflow.switchPhase(phaseId);
    }),
    vscode.commands.registerCommand('sparc-artifacts.openArtifact', (path) => {
      vscode.workspace.openTextDocument(path).then(doc => {
        vscode.window.showTextDocument(doc);
      });
    })
  );
  
  context.subscriptions.push(treeView);
}
```

## AI-Powered Extensions

SPARC IDE provides APIs for creating AI-powered extensions:

### AI Integration API

The AI Integration API allows you to work with multiple AI models:

```typescript
interface AIIntegrationAPI {
  // Model management
  getCurrentModel(): Promise<AIModel>;
  switchModel(modelId: string): Promise<AIModel>;
  getModels(): Promise<AIModel[]>;
  
  // Mode management
  getCurrentMode(): Promise<AIMode>;
  switchMode(modeId: string): Promise<AIMode>;
  getModes(): Promise<AIMode[]>;
  
  // Prompt execution
  executePrompt(prompt: string, options?: PromptOptions): Promise<PromptResult>;
  
  // Multi-agent workflow
  executeMultiAgentWorkflow(task: string, agents?: Agent[]): Promise<WorkflowResult>;
}
```

### Example: Creating an AI-Powered Code Reviewer

Here's an example of how to create an extension that uses AI to review code:

```typescript
import * as vscode from 'vscode';

export async function activate(context: vscode.ExtensionContext) {
  // Access the AI Integration API
  const aiIntegration = vscode.extensions.getExtension('sparc-ide.ai-integration')?.exports;
  
  if (!aiIntegration) {
    vscode.window.showErrorMessage('AI Integration API not available');
    return;
  }
  
  // Register a command to review code
  let disposable = vscode.commands.registerCommand('ai-code-reviewer.reviewCode', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('No active editor');
      return;
    }
    
    const document = editor.document;
    const code = document.getText();
    const language = document.languageId;
    
    // Show progress indicator
    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: 'Reviewing code...',
      cancellable: true
    }, async (progress, token) => {
      // Create the prompt for code review
      const prompt = `
        Review the following ${language} code for:
        1. Bugs and logical errors
        2. Performance issues
        3. Security vulnerabilities
        4. Style and best practices
        5. Potential improvements
        
        Provide specific feedback with line numbers and suggested fixes.
        
        Code to review:
        \`\`\`${language}
        ${code}
        \`\`\`
      `;
      
      // Execute the prompt with the Code Reviewer mode
      await aiIntegration.switchMode('code-reviewer');
      const result = await aiIntegration.executePrompt(prompt, {
        temperature: 0.3,
        maxTokens: 2000
      });
      
      // Create a new file with the review results
      const reviewUri = document.uri.with({ path: document.uri.path + '.review.md' });
      const edit = new vscode.WorkspaceEdit();
      edit.createFile(reviewUri, { overwrite: true });
      
      // Format the review results
      const reviewContent = `# Code Review: ${document.uri.path.split('/').pop()}\n\n${result.text}`;
      edit.insert(reviewUri, new vscode.Position(0, 0), reviewContent);
      
      await vscode.workspace.applyEdit(edit);
      await vscode.window.showTextDocument(reviewUri);
    });
  });
  
  context.subscriptions.push(disposable);
}
```

## Testing and Debugging

Testing and debugging are essential parts of extension development:

### Unit Testing

You can write unit tests for your extension using the Mocha framework:

```typescript
import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Extension Test Suite', () => {
  vscode.window.showInformationMessage('Starting tests...');

  test('Sample test', () => {
    assert.strictEqual(1 + 1, 2);
  });
  
  test('Extension activation', async () => {
    const extension = vscode.extensions.getExtension('your-publisher.your-extension');
    assert.ok(extension);
    
    await extension?.activate();
    assert.strictEqual(extension?.isActive, true);
  });
  
  test('Command execution', async () => {
    // Mock the showInformationMessage function
    const originalShowInformationMessage = vscode.window.showInformationMessage;
    let messageShown = false;
    vscode.window.showInformationMessage = async (message: string) => {
      if (message === 'Hello from SPARC IDE!') {
        messageShown = true;
      }
      return undefined as any;
    };
    
    // Execute the command
    await vscode.commands.executeCommand('your-extension.helloWorld');
    
    // Restore the original function
    vscode.window.showInformationMessage = originalShowInformationMessage;
    
    // Assert that the message was shown
    assert.strictEqual(messageShown, true);
  });
});
```

### Integration Testing

You can write integration tests that launch a new instance of SPARC IDE:

```typescript
import * as path from 'path';
import * as Mocha from 'mocha';
import * as glob from 'glob';
import { runTests } from 'vscode-test';

async function main() {
  try {
    // The folder containing the Extension Manifest package.json
    const extensionDevelopmentPath = path.resolve(__dirname, '../../');
    
    // The path to the extension test script
    const extensionTestsPath = path.resolve(__dirname, './suite/index');
    
    // The path to a workspace to open for the test
    const testWorkspacePath = path.resolve(__dirname, '../../test-workspace');
    
    // Download VS Code, unzip it, and run the integration tests
    await runTests({
      extensionDevelopmentPath,
      extensionTestsPath,
      launchArgs: [testWorkspacePath]
    });
  } catch (err) {
    console.error('Failed to run tests');
    process.exit(1);
  }
}

main();
```

### Debugging

To debug your extension:

1. Set breakpoints in your code
2. Press F5 to launch a new SPARC IDE instance with your extension
3. Trigger the functionality you want to debug
4. The debugger will stop at your breakpoints
5. Inspect variables, step through code, and diagnose issues

## Publishing Your Extension

Once your extension is ready, you can publish it for others to use:

### Packaging Your Extension

1. Install the VS Code Extension Manager:
   ```bash
   npm install -g vsce
   ```

2. Package your extension:
   ```bash
   vsce package
   ```

   This creates a `.vsix` file that can be installed in SPARC IDE.

### Testing the Packaged Extension

1. Install the packaged extension in SPARC IDE:
   ```bash
   sparc-ide --install-extension your-extension-0.0.1.vsix
   ```

2. Test that your extension works correctly

### Publishing to the VS Code Marketplace

1. Create a publisher account on the [VS Code Marketplace](https://marketplace.visualstudio.com/VSCode)

2. Get a Personal Access Token (PAT) from Azure DevOps

3. Login with your PAT:
   ```bash
   vsce login <publisher-name>
   ```

4. Publish your extension:
   ```bash
   vsce publish
   ```

### Publishing to the Open VSX Registry

For open-source distribution, you can publish to the Open VSX Registry:

1. Create an account on the [Open VSX Registry](https://open-vsx.org/)

2. Get a PAT from Open VSX

3. Publish your extension:
   ```bash
   npx ovsx publish -p <pat>
   ```

### Maintaining Your Extension

1. Update your extension regularly:
   - Fix bugs
   - Add new features
   - Keep dependencies up to date

2. Increment the version number in `package.json`

3. Update the changelog

4. Publish the new version:
   ```bash
   vsce publish
   ```

5. Respond to user feedback and issues