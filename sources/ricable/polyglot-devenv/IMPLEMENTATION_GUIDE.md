# Swarm Flow Implementation Guide

## Phase-by-Phase Implementation Strategy

This guide provides detailed, actionable steps for implementing the Swarm Flow refactoring. Each phase includes specific code examples, migration scripts, and validation steps.

## Pre-Implementation Setup

### 1. Environment Preparation
```bash
# Install required tools
npm install -g pnpm@latest
pnpm --version  # Should be 8.0.0+

# Create backup of current system
cp -r claude-flow backups/claude-flow-pre-refactor-$(date +%Y%m%d)
```

### 2. Dependency Analysis
Before starting, we need to understand the current dependencies:

```bash
# Analyze current claude-flow dependencies
cd claude-flow
npm ls --depth=0 > ../analysis/current-dependencies.txt

# Identify shared dependencies across projects
cd ../mcp
npm ls --depth=0 > ../analysis/mcp-dependencies.txt
```

## Phase 1: Monorepo Foundation (Days 1-3)

### Step 1.1: Create Root Structure
```bash
# Create new directory structure
mkdir -p packages/{sdk,common,memory,orchestrator,provider-devpod,mcp-integration}
mkdir -p apps/{cli,docs,web-ui}
mkdir -p examples/{01-simple-task,02-rest-api-generation,03-polyglot-swarm}
```

### Step 1.2: Root Package Configuration
Create `package.json`:
```json
{
  "name": "@swarm-flow/workspace",
  "version": "2.0.0",
  "private": true,
  "type": "module",
  "packageManager": "pnpm@8.15.0",
  "engines": {
    "node": ">=18.0.0",
    "pnpm": ">=8.0.0"
  },
  "scripts": {
    "build": "pnpm -r build",
    "build:watch": "pnpm -r --parallel build:watch",
    "test": "pnpm -r test",
    "test:watch": "pnpm -r --parallel test:watch",
    "lint": "pnpm -r lint",
    "lint:fix": "pnpm -r lint:fix",
    "format": "pnpm -r format",
    "format:check": "pnpm -r format:check",
    "typecheck": "pnpm -r typecheck",
    "clean": "pnpm -r clean && rm -rf node_modules",
    "dev": "pnpm -r --parallel dev",
    "changeset": "changeset",
    "version-packages": "changeset version",
    "release": "pnpm build && changeset publish"
  },
  "devDependencies": {
    "@changesets/cli": "^2.27.1",
    "@typescript-eslint/eslint-plugin": "^6.15.0",
    "@typescript-eslint/parser": "^6.15.0",
    "eslint": "^8.56.0",
    "eslint-config-prettier": "^9.1.0",
    "prettier": "^3.1.1",
    "typescript": "^5.3.3",
    "tsx": "^4.6.2"
  }
}
```

### Step 1.3: Workspace Configuration
Create `pnpm-workspace.yaml`:
```yaml
packages:
  - 'packages/*'
  - 'apps/*'
  - 'examples/*'
```

### Step 1.4: Base TypeScript Configuration
Create `tsconfig.base.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "node",
    "allowSyntheticDefaultImports": true,
    "esModuleInterop": true,
    "allowJs": true,
    "checkJs": false,
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noImplicitOverride": true,
    "exactOptionalPropertyTypes": false,
    "noPropertyAccessFromIndexSignature": false,
    "noUncheckedIndexedAccess": false,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": false,
    "skipLibCheck": true,
    "composite": true,
    "incremental": true,
    "lib": ["ES2022", "DOM"],
    "types": ["node"]
  },
  "exclude": ["node_modules", "dist", "**/*.test.ts", "**/*.spec.ts"]
}
```

### Step 1.5: Shared Tooling Configuration
Create `.eslintrc.base.js`:
```javascript
module.exports = {
  parser: '@typescript-eslint/parser',
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'prettier'
  ],
  plugins: ['@typescript-eslint'],
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module'
  },
  rules: {
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/explicit-function-return-type': 'warn',
    '@typescript-eslint/no-explicit-any': 'warn',
    'prefer-const': 'error',
    'no-var': 'error'
  }
};
```

Create `.prettierrc.json`:
```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false
}
```

## Phase 2: Core Package Migration (Days 4-10)

### Step 2.1: Create @swarm-flow/common Package

#### Package Structure
```bash
cd packages/common
```

Create `package.json`:
```json
{
  "name": "@swarm-flow/common",
  "version": "2.0.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js"
    }
  },
  "files": ["dist"],
  "scripts": {
    "build": "tsc",
    "build:watch": "tsc --watch",
    "test": "jest",
    "test:watch": "jest --watch",
    "lint": "eslint src --ext .ts",
    "lint:fix": "eslint src --ext .ts --fix",
    "format": "prettier --write src",
    "format:check": "prettier --check src",
    "typecheck": "tsc --noEmit",
    "clean": "rm -rf dist"
  },
  "dependencies": {
    "chalk": "^5.3.0",
    "fs-extra": "^11.2.0",
    "nanoid": "^5.0.4"
  },
  "devDependencies": {
    "@types/fs-extra": "^11.0.4",
    "@types/jest": "^29.5.8",
    "@types/node": "^20.10.5",
    "jest": "^29.7.0",
    "ts-jest": "^29.1.1"
  }
}
```

#### Migration Script for Common Utilities
Create `migrate-common.ts`:
```typescript
import { promises as fs } from 'fs';
import path from 'path';

interface FileMapping {
  source: string;
  destination: string;
  transform?: (content: string) => string;
}

const fileMappings: FileMapping[] = [
  {
    source: 'claude-flow/src/utils/logger.ts',
    destination: 'packages/common/src/logger.ts',
    transform: (content) => content.replace(/from '\.\.\//, "from './")
  },
  {
    source: 'claude-flow/src/utils/filesystem.ts',
    destination: 'packages/common/src/filesystem.ts'
  },
  {
    source: 'claude-flow/src/utils/process.ts',
    destination: 'packages/common/src/process.ts'
  },
  {
    source: 'claude-flow/src/config/',
    destination: 'packages/common/src/config/'
  }
];

async function migrateCommonFiles(): Promise<void> {
  for (const mapping of fileMappings) {
    const sourcePath = path.resolve(mapping.source);
    const destPath = path.resolve(mapping.destination);
    
    try {
      const content = await fs.readFile(sourcePath, 'utf-8');
      const transformedContent = mapping.transform ? mapping.transform(content) : content;
      
      await fs.mkdir(path.dirname(destPath), { recursive: true });
      await fs.writeFile(destPath, transformedContent);
      
      console.log(`✅ Migrated: ${mapping.source} → ${mapping.destination}`);
    } catch (error) {
      console.error(`❌ Failed to migrate ${mapping.source}:`, error);
    }
  }
}

migrateCommonFiles().catch(console.error);
```

#### Core Common Utilities
Create `packages/common/src/logger.ts`:
```typescript
import chalk from 'chalk';

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

export interface LoggerConfig {
  level: LogLevel;
  prefix?: string;
  timestamp?: boolean;
}

export class Logger {
  private config: LoggerConfig;

  constructor(config: LoggerConfig = { level: LogLevel.INFO }) {
    this.config = config;
  }

  debug(message: string, ...args: any[]): void {
    if (this.config.level <= LogLevel.DEBUG) {
      this.log(chalk.gray('[DEBUG]'), message, ...args);
    }
  }

  info(message: string, ...args: any[]): void {
    if (this.config.level <= LogLevel.INFO) {
      this.log(chalk.blue('[INFO]'), message, ...args);
    }
  }

  warn(message: string, ...args: any[]): void {
    if (this.config.level <= LogLevel.WARN) {
      this.log(chalk.yellow('[WARN]'), message, ...args);
    }
  }

  error(message: string, ...args: any[]): void {
    if (this.config.level <= LogLevel.ERROR) {
      this.log(chalk.red('[ERROR]'), message, ...args);
    }
  }

  private log(level: string, message: string, ...args: any[]): void {
    const timestamp = this.config.timestamp ? `[${new Date().toISOString()}] ` : '';
    const prefix = this.config.prefix ? `[${this.config.prefix}] ` : '';
    console.log(`${timestamp}${prefix}${level} ${message}`, ...args);
  }
}

export const createLogger = (config?: LoggerConfig): Logger => new Logger(config);
```

### Step 2.2: Create @swarm-flow/sdk Package

Create `packages/sdk/src/types/core.ts`:
```typescript
// Core type definitions
export type AgentId = string;
export type TaskId = string;
export type SwarmId = string;
export type EnvironmentId = string;

export enum AgentType {
  CODER = 'coder',
  TESTER = 'tester',
  REVIEWER = 'reviewer',
  COORDINATOR = 'coordinator',
  SPECIALIST = 'specialist',
}

export enum TaskType {
  CODE_GENERATION = 'code_generation',
  CODE_REVIEW = 'code_review',
  TESTING = 'testing',
  DOCUMENTATION = 'documentation',
  REFACTORING = 'refactoring',
  ANALYSIS = 'analysis',
}

export enum TaskStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export enum Priority {
  LOW = 1,
  MEDIUM = 2,
  HIGH = 3,
  CRITICAL = 4,
}

export interface Capability {
  name: string;
  version: string;
  description: string;
  parameters?: Record<string, any>;
}

export interface TaskRequirement {
  type: string;
  value: any;
  optional?: boolean;
}

export interface TaskConstraint {
  type: string;
  value: any;
  strict?: boolean;
}
```

Create `packages/sdk/src/interfaces/agent.ts`:
```typescript
import { AgentId, AgentType, Capability, TaskId } from '../types/core.js';
import { Task, TaskResult } from './task.js';
import { ExecutionContext } from './execution.js';

export interface Agent {
  readonly id: AgentId;
  readonly name: string;
  readonly type: AgentType;
  readonly capabilities: Capability[];
  readonly environment: string;
  
  execute(task: Task, context: ExecutionContext): Promise<TaskResult>;
  getStatus(): Promise<AgentStatus>;
  terminate(): Promise<void>;
}

export interface AgentStatus {
  id: AgentId;
  state: AgentState;
  currentTask?: TaskId;
  uptime: number;
  metrics: AgentMetrics;
}

export enum AgentState {
  IDLE = 'idle',
  BUSY = 'busy',
  ERROR = 'error',
  TERMINATED = 'terminated',
}

export interface AgentMetrics {
  tasksCompleted: number;
  averageExecutionTime: number;
  successRate: number;
  lastActivity: Date;
}

export interface AgentConfig {
  name: string;
  type: AgentType;
  capabilities: Capability[];
  environment: string;
  resources?: ResourceRequirements;
}

export interface ResourceRequirements {
  cpu?: number;
  memory?: string;
  storage?: string;
}

export interface AgentFactory {
  createAgent(config: AgentConfig): Promise<Agent>;
  getAvailableTypes(): AgentType[];
  getSupportedCapabilities(): Capability[];
}
```

### Step 2.3: Create Migration Validation Script

Create `scripts/validate-migration.ts`:
```typescript
import { promises as fs } from 'fs';
import path from 'path';
import { execSync } from 'child_process';

interface ValidationResult {
  package: string;
  builds: boolean;
  tests: boolean;
  types: boolean;
  errors: string[];
}

async function validatePackage(packagePath: string): Promise<ValidationResult> {
  const packageName = path.basename(packagePath);
  const result: ValidationResult = {
    package: packageName,
    builds: false,
    tests: false,
    types: false,
    errors: []
  };

  try {
    // Check if package.json exists
    const packageJsonPath = path.join(packagePath, 'package.json');
    await fs.access(packageJsonPath);

    // Try to build
    try {
      execSync('pnpm build', { cwd: packagePath, stdio: 'pipe' });
      result.builds = true;
    } catch (error) {
      result.errors.push(`Build failed: ${error}`);
    }

    // Try to run tests
    try {
      execSync('pnpm test', { cwd: packagePath, stdio: 'pipe' });
      result.tests = true;
    } catch (error) {
      result.errors.push(`Tests failed: ${error}`);
    }

    // Check TypeScript compilation
    try {
      execSync('pnpm typecheck', { cwd: packagePath, stdio: 'pipe' });
      result.types = true;
    } catch (error) {
      result.errors.push(`Type checking failed: ${error}`);
    }

  } catch (error) {
    result.errors.push(`Package validation failed: ${error}`);
  }

  return result;
}

async function validateAllPackages(): Promise<void> {
  const packagesDir = path.resolve('packages');
  const packages = await fs.readdir(packagesDir);
  
  console.log('🔍 Validating migrated packages...\n');
  
  const results: ValidationResult[] = [];
  
  for (const pkg of packages) {
    const packagePath = path.join(packagesDir, pkg);
    const stat = await fs.stat(packagePath);
    
    if (stat.isDirectory()) {
      const result = await validatePackage(packagePath);
      results.push(result);
      
      const status = result.builds && result.tests && result.types ? '✅' : '❌';
      console.log(`${status} ${result.package}`);
      
      if (result.errors.length > 0) {
        result.errors.forEach(error => console.log(`   ⚠️  ${error}`));
      }
    }
  }
  
  console.log('\n📊 Validation Summary:');
  const successful = results.filter(r => r.builds && r.tests && r.types).length;
  console.log(`✅ Successful: ${successful}/${results.length}`);
  
  if (successful < results.length) {
    console.log('❌ Some packages failed validation. Please fix errors before proceeding.');
    process.exit(1);
  } else {
    console.log('🎉 All packages validated successfully!');
  }
}

validateAllPackages().catch(console.error);
```

## Phase 3: Provider and Orchestrator (Days 11-17)

### Step 3.1: DevPod Provider Implementation

Create `packages/provider-devpod/src/devpod-provider.ts`:
```typescript
import { EnvironmentProvider, Environment, EnvironmentConfig } from '@swarm-flow/sdk';
import { Logger } from '@swarm-flow/common';
import { spawn } from 'child_process';
import { promisify } from 'util';

export class DevPodProvider implements EnvironmentProvider {
  private logger = new Logger({ prefix: 'DevPodProvider' });

  async provision(config: EnvironmentConfig): Promise<Environment> {
    this.logger.info(`Provisioning environment: ${config.name}`);
    
    const workspaceName = this.generateWorkspaceName(config);
    
    try {
      // Create DevPod workspace
      await this.executeDevPodCommand([
        'up',
        workspaceName,
        '--source', this.getSourcePath(config),
        '--devcontainer-path', this.getDevcontainerPath(config.language),
        '--ide', 'none' // We'll manage IDE separately
      ]);

      // Wait for workspace to be ready
      await this.waitForWorkspaceReady(workspaceName);

      // Setup environment-specific configuration
      await this.setupEnvironment(workspaceName, config);

      return new DevPodEnvironment(workspaceName, this);
    } catch (error) {
      this.logger.error(`Failed to provision environment: ${error}`);
      throw error;
    }
  }

  async destroy(environmentId: string): Promise<void> {
    this.logger.info(`Destroying environment: ${environmentId}`);
    
    try {
      await this.executeDevPodCommand(['delete', environmentId, '--force']);
    } catch (error) {
      this.logger.error(`Failed to destroy environment: ${error}`);
      throw error;
    }
  }

  async status(environmentId: string): Promise<EnvironmentStatus> {
    try {
      const output = await this.executeDevPodCommand(['status', environmentId, '--output', 'json']);
      return this.parseDevPodStatus(output);
    } catch (error) {
      this.logger.error(`Failed to get status: ${error}`);
      throw error;
    }
  }

  private generateWorkspaceName(config: EnvironmentConfig): string {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    return `swarm-${config.language}-${timestamp}`;
  }

  private getDevcontainerPath(language: string): string {
    const devcontainerMap = {
      python: 'dev-env/python/.devcontainer',
      typescript: 'dev-env/typescript/.devcontainer',
      rust: 'dev-env/rust/.devcontainer',
      go: 'dev-env/go/.devcontainer',
      nushell: 'dev-env/nushell/.devcontainer'
    };
    
    return devcontainerMap[language] || devcontainerMap.python;
  }

  private async executeDevPodCommand(args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const process = spawn('devpod', args, { stdio: 'pipe' });
      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(`DevPod command failed: ${stderr}`));
        }
      });
    });
  }

  private async waitForWorkspaceReady(workspaceName: string, timeout = 300000): Promise<void> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      try {
        const status = await this.status(workspaceName);
        if (status.state === 'Running') {
          return;
        }
      } catch (error) {
        // Workspace might not exist yet, continue waiting
      }
      
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
    
    throw new Error(`Workspace ${workspaceName} failed to become ready within timeout`);
  }
}
```

### Step 3.2: Orchestrator Core Implementation

Create `packages/orchestrator/src/swarm-orchestrator.ts`:
```typescript
import { 
  Task, TaskResult, Swarm, SwarmResult, 
  EnvironmentProvider, MemoryProvider, AgentFactory 
} from '@swarm-flow/sdk';
import { Logger } from '@swarm-flow/common';

export class SwarmOrchestrator {
  private logger = new Logger({ prefix: 'SwarmOrchestrator' });

  constructor(
    private environmentProvider: EnvironmentProvider,
    private memoryProvider: MemoryProvider,
    private agentFactory: AgentFactory
  ) {}

  async executeTask(task: Task): Promise<TaskResult> {
    this.logger.info(`Executing task: ${task.id}`);
    
    try {
      // Create optimal swarm for the task
      const swarm = await this.createOptimalSwarm(task);
      
      // Plan execution strategy
      const executionPlan = await this.planExecution(task, swarm);
      
      // Execute with monitoring
      const result = await this.executeWithMonitoring(executionPlan);
      
      // Store result in memory
      await this.memoryProvider.store(`task:${task.id}`, result);
      
      return result;
    } catch (error) {
      this.logger.error(`Task execution failed: ${error}`);
      throw error;
    }
  }

  private async createOptimalSwarm(task: Task): Promise<Swarm> {
    // Analyze task requirements
    const requiredCapabilities = this.analyzeTaskRequirements(task);
    
    // Select optimal agents
    const agents = await this.selectOptimalAgents(requiredCapabilities);
    
    // Determine execution strategy
    const strategy = this.determineExecutionStrategy(task, agents);
    
    // Create swarm
    return {
      id: this.generateSwarmId(),
      name: `swarm-${task.id}`,
      agents,
      strategy,
      coordinator: await this.createCoordinator(strategy)
    };
  }

  private analyzeTaskRequirements(task: Task): string[] {
    const capabilities: string[] = [];
    
    // Analyze task type
    switch (task.type) {
      case 'code_generation':
        capabilities.push('coding', 'language-specific');
        break;
      case 'code_review':
        capabilities.push('review', 'quality-analysis');
        break;
      case 'testing':
        capabilities.push('testing', 'validation');
        break;
      case 'documentation':
        capabilities.push('documentation', 'writing');
        break;
    }
    
    // Analyze task requirements
    task.requirements.forEach(req => {
      if (req.type === 'language') {
        capabilities.push(`language-${req.value}`);
      }
      if (req.type === 'framework') {
        capabilities.push(`framework-${req.value}`);
      }
    });
    
    return capabilities;
  }

  private async selectOptimalAgents(requiredCapabilities: string[]): Promise<Agent[]> {
    const availableAgents = await this.agentFactory.getAvailableAgents();
    const selectedAgents: Agent[] = [];
    
    for (const capability of requiredCapabilities) {
      const bestAgent = availableAgents.find(agent => 
        agent.capabilities.some(cap => cap.name === capability)
      );
      
      if (bestAgent && !selectedAgents.includes(bestAgent)) {
        selectedAgents.push(bestAgent);
      }
    }
    
    // Ensure we have at least one agent
    if (selectedAgents.length === 0) {
      const defaultAgent = await this.agentFactory.createAgent({
        name: 'default-coder',
        type: 'coder',
        capabilities: [{ name: 'general-coding', version: '1.0.0', description: 'General coding capabilities' }],
        environment: 'python'
      });
      selectedAgents.push(defaultAgent);
    }
    
    return selectedAgents;
  }
}
```

## Phase 4: CLI Application (Days 18-22)

### Step 4.1: Modern CLI with oclif

Create `apps/cli/package.json`:
```json
{
  "name": "@swarm-flow/cli",
  "version": "2.0.0",
  "type": "module",
  "bin": {
    "swarm-flow": "./bin/run.js"
  },
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "files": ["bin", "dist"],
  "scripts": {
    "build": "tsc",
    "dev": "tsx src/index.ts",
    "test": "jest",
    "lint": "eslint src --ext .ts",
    "format": "prettier --write src"
  },
  "dependencies": {
    "@oclif/core": "^3.15.1",
    "@oclif/plugin-help": "^6.0.8",
    "@oclif/plugin-plugins": "^4.1.8",
    "@swarm-flow/sdk": "workspace:*",
    "@swarm-flow/orchestrator": "workspace:*",
    "@swarm-flow/provider-devpod": "workspace:*",
    "@swarm-flow/common": "workspace:*",
    "chalk": "^5.3.0",
    "inquirer": "^9.2.12",
    "ora": "^7.0.1"
  },
  "devDependencies": {
    "@types/inquirer": "^9.0.7"
  },
  "oclif": {
    "bin": "swarm-flow",
    "dirname": "swarm-flow",
    "commands": "./dist/commands",
    "plugins": ["@oclif/plugin-help", "@oclif/plugin-plugins"],
    "topicSeparator": " "
  }
}
```

Create `apps/cli/src/commands/task/create.ts`:
```typescript
import { Command, Flags } from '@oclif/core';
import { SwarmOrchestrator } from '@swarm-flow/orchestrator';
import { DevPodProvider } from '@swarm-flow/provider-devpod';
import { HybridMemoryProvider } from '@swarm-flow/memory';
import { AgentFactoryImpl } from '@swarm-flow/orchestrator';
import { Task, TaskType, Priority } from '@swarm-flow/sdk';
import { Logger } from '@swarm-flow/common';
import chalk from 'chalk';
import ora from 'ora';

export default class TaskCreate extends Command {
  static description = 'Create and execute a new task';

  static examples = [
    '$ swarm-flow task create "Build a REST API for user management"',
    '$ swarm-flow task create "Refactor authentication module" --language python --priority high',
    '$ swarm-flow task create "Add unit tests" --type testing --environment typescript'
  ];

  static flags = {
    help: Flags.help({ char: 'h' }),
    language: Flags.string({
      char: 'l',
      description: 'Target programming language',
      options: ['python', 'typescript', 'rust', 'go', 'nushell'],
      default: 'python'
    }),
    type: Flags.string({
      char: 't',
      description: 'Task type',
      options: ['code_generation', 'code_review', 'testing', 'documentation', 'refactoring', 'analysis'],
      default: 'code_generation'
    }),
    priority: Flags.string({
      char: 'p',
      description: 'Task priority',
      options: ['low', 'medium', 'high', 'critical'],
      default: 'medium'
    }),
    environment: Flags.string({
      char: 'e',
      description: 'Target environment',
      default: 'auto'
    }),
    agents: Flags.integer({
      char: 'a',
      description: 'Number of agents to use',
      default: 1,
      min: 1,
      max: 10
    }),
    strategy: Flags.string({
      char: 's',
      description: 'Execution strategy',
      options: ['parallel', 'sequential', 'hierarchical', 'collaborative'],
      default: 'parallel'
    }),
    dry_run: Flags.boolean({
      char: 'd',
      description: 'Show execution plan without running',
      default: false
    })
  };

  static args = [
    {
      name: 'description',
      description: 'Task description',
      required: true
    }
  ];

  async run(): Promise<void> {
    const { args, flags } = await this.parse(TaskCreate);
    const logger = new Logger({ prefix: 'TaskCreate' });

    // Create task
    const task: Task = {
      id: this.generateTaskId(),
      type: flags.type as TaskType,
      description: args.description,
      requirements: [
        { type: 'language', value: flags.language },
        { type: 'agents', value: flags.agents },
        { type: 'strategy', value: flags.strategy }
      ],
      constraints: [],
      priority: this.mapPriority(flags.priority),
      dependencies: []
    };

    // Initialize orchestrator
    const orchestrator = new SwarmOrchestrator(
      new DevPodProvider(),
      new HybridMemoryProvider