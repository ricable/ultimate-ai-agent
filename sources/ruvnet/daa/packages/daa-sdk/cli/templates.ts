/**
 * Template Management Utilities
 *
 * Functions for scaffolding new DAA projects from templates
 */

import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import { exec } from 'child_process';

const execAsync = promisify(exec);
const copyFile = promisify(fs.copyFile);
const mkdir = promisify(fs.mkdir);
const readdir = promisify(fs.readdir);
const stat = promisify(fs.stat);
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

export interface TemplateOptions {
  name: string;
  template: 'basic' | 'full-stack' | 'ml-training';
  typescript: boolean;
  installDeps: boolean;
  gitInit: boolean;
}

export interface TemplateInfo {
  name: string;
  description: string;
  features: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

/**
 * Available project templates
 */
export const TEMPLATES: Record<string, TemplateInfo> = {
  basic: {
    name: 'Basic',
    description: 'Simple DAA agent with quantum-resistant cryptography',
    features: [
      'ML-KEM-768 key encapsulation',
      'ML-DSA digital signatures',
      'BLAKE3 hashing',
      'Password vault',
    ],
    difficulty: 'beginner',
  },
  'full-stack': {
    name: 'Full-Stack',
    description: 'Complete DAA ecosystem with orchestrator and workflows',
    features: [
      'MRAP orchestrator',
      'Workflow engine',
      'QuDAG networking',
      'Rules engine',
      'Token economy',
    ],
    difficulty: 'intermediate',
  },
  'ml-training': {
    name: 'ML Training',
    description: 'Federated machine learning with privacy-preserving training',
    features: [
      'Federated learning',
      'Differential privacy',
      'Secure aggregation',
      'Multiple architectures',
      'Distributed inference',
    ],
    difficulty: 'advanced',
  },
};

/**
 * Create a new project from a template
 */
export async function scaffoldProject(options: TemplateOptions): Promise<void> {
  const { name, template } = options;

  // Validate template
  if (!TEMPLATES[template]) {
    throw new Error(`Unknown template: ${template}`);
  }

  // Create project directory
  const projectPath = path.join(process.cwd(), name);
  await createDirectory(projectPath);

  // Get template source path
  const templatePath = path.join(__dirname, '..', 'templates', template);

  // Copy template files
  await copyDirectory(templatePath, projectPath);

  // Update package.json with project name
  await updatePackageJson(projectPath, name);

  // Initialize git if requested
  if (options.gitInit) {
    await initGit(projectPath);
  }

  // Install dependencies if requested
  if (options.installDeps) {
    await installDependencies(projectPath);
  }
}

/**
 * Create directory recursively
 */
async function createDirectory(dirPath: string): Promise<void> {
  if (fs.existsSync(dirPath)) {
    throw new Error(`Directory already exists: ${dirPath}`);
  }
  await mkdir(dirPath, { recursive: true });
}

/**
 * Copy directory recursively
 */
async function copyDirectory(src: string, dest: string): Promise<void> {
  await mkdir(dest, { recursive: true });

  const entries = await readdir(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      await copyDirectory(srcPath, destPath);
    } else {
      await copyFile(srcPath, destPath);
    }
  }
}

/**
 * Update package.json with project name
 */
async function updatePackageJson(projectPath: string, projectName: string): Promise<void> {
  const packageJsonPath = path.join(projectPath, 'package.json');

  if (!fs.existsSync(packageJsonPath)) {
    return;
  }

  const content = await readFile(packageJsonPath, 'utf-8');
  const packageJson = JSON.parse(content);

  // Update name
  packageJson.name = projectName;

  // Write back
  await writeFile(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n');
}

/**
 * Initialize git repository
 */
async function initGit(projectPath: string): Promise<void> {
  try {
    await execAsync('git init', { cwd: projectPath });
    await execAsync('git add .', { cwd: projectPath });
    await execAsync('git commit -m "Initial commit from DAA SDK template"', {
      cwd: projectPath,
    });
  } catch (error) {
    // Git initialization is optional, don't fail if it errors
    console.warn('⚠️  Git initialization failed (is git installed?)');
  }
}

/**
 * Install npm dependencies
 */
async function installDependencies(projectPath: string): Promise<void> {
  try {
    console.log('\n📦 Installing dependencies...');
    await execAsync('npm install', { cwd: projectPath });
    console.log('✅ Dependencies installed');
  } catch (error: any) {
    throw new Error(`Failed to install dependencies: ${error.message}`);
  }
}

/**
 * Get template information
 */
export function getTemplateInfo(templateName: string): TemplateInfo | null {
  return TEMPLATES[templateName] || null;
}

/**
 * List all available templates
 */
export function listTemplates(): TemplateInfo[] {
  return Object.values(TEMPLATES);
}

/**
 * Check if directory exists and is empty
 */
export async function isDirectoryEmpty(dirPath: string): Promise<boolean> {
  if (!fs.existsSync(dirPath)) {
    return true;
  }

  const entries = await readdir(dirPath);
  return entries.length === 0;
}

/**
 * Validate project name
 */
export function validateProjectName(name: string): { valid: boolean; error?: string } {
  // Check for empty name
  if (!name || name.trim().length === 0) {
    return { valid: false, error: 'Project name cannot be empty' };
  }

  // Check for invalid characters
  if (!/^[a-z0-9-_]+$/i.test(name)) {
    return {
      valid: false,
      error: 'Project name can only contain letters, numbers, hyphens, and underscores',
    };
  }

  // Check for reserved names
  const reserved = ['node_modules', 'test', 'tests', '.git'];
  if (reserved.includes(name.toLowerCase())) {
    return { valid: false, error: `'${name}' is a reserved name` };
  }

  return { valid: true };
}

/**
 * Generate project summary
 */
export function generateProjectSummary(options: TemplateOptions): string {
  const template = TEMPLATES[options.template];

  return `
📋 Project Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Name: ${options.name}
Template: ${template.name}
Language: ${options.typescript ? 'TypeScript' : 'JavaScript'}
Git: ${options.gitInit ? 'Initialized' : 'Not initialized'}
Dependencies: ${options.installDeps ? 'Installed' : 'Not installed'}

Features:
${template.features.map((f) => `  • ${f}`).join('\n')}

Difficulty: ${template.difficulty}

Next Steps:
  1. cd ${options.name}
  ${options.installDeps ? '' : '2. npm install\n  '}${options.installDeps ? '2' : '3'}. npm run dev

Happy coding! 🚀
`;
}
