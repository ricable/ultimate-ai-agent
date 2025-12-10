/**
 * Interactive CLI Prompts
 *
 * Utilities for interactive project setup
 */

import * as readline from 'readline';

export interface PromptOptions {
  message: string;
  defaultValue?: string;
  choices?: string[];
  validate?: (input: string) => boolean | string;
}

/**
 * Create readline interface
 */
function createInterface(): readline.Interface {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
}

/**
 * Prompt user for input
 */
export function prompt(options: PromptOptions): Promise<string> {
  return new Promise((resolve) => {
    const rl = createInterface();

    const defaultText = options.defaultValue ? ` (${options.defaultValue})` : '';
    const message = `${options.message}${defaultText}: `;

    rl.question(message, (answer) => {
      rl.close();

      const input = answer.trim() || options.defaultValue || '';

      // Validate input
      if (options.validate) {
        const validation = options.validate(input);
        if (validation !== true) {
          console.log(`❌ ${validation}`);
          return resolve(prompt(options));
        }
      }

      resolve(input);
    });
  });
}

/**
 * Prompt user to select from choices
 */
export async function select(options: PromptOptions): Promise<string> {
  if (!options.choices || options.choices.length === 0) {
    throw new Error('Choices are required for select prompt');
  }

  console.log(`\n${options.message}`);
  options.choices.forEach((choice, i) => {
    const indicator = choice === options.defaultValue ? '→' : ' ';
    console.log(`  ${indicator} [${i + 1}] ${choice}`);
  });
  console.log();

  const answer = await prompt({
    message: 'Select an option',
    defaultValue: '1',
    validate: (input) => {
      const num = parseInt(input, 10);
      if (isNaN(num) || num < 1 || num > options.choices!.length) {
        return `Please enter a number between 1 and ${options.choices!.length}`;
      }
      return true;
    },
  });

  const index = parseInt(answer, 10) - 1;
  return options.choices[index];
}

/**
 * Prompt user for yes/no confirmation
 */
export async function confirm(message: string, defaultValue: boolean = true): Promise<boolean> {
  const defaultText = defaultValue ? 'Y/n' : 'y/N';
  const answer = await prompt({
    message: `${message} (${defaultText})`,
    defaultValue: defaultValue ? 'y' : 'n',
  });

  return answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes';
}

/**
 * Interactive project setup
 */
export async function interactiveSetup(): Promise<{
  name: string;
  template: string;
  typescript: boolean;
  installDeps: boolean;
  gitInit: boolean;
}> {
  console.log('\n🚀 DAA SDK Project Setup\n');

  // Project name
  const name = await prompt({
    message: 'Project name',
    validate: (input) => {
      if (!input || input.trim().length === 0) {
        return 'Project name is required';
      }
      if (!/^[a-z0-9-_]+$/i.test(input)) {
        return 'Project name can only contain letters, numbers, hyphens, and underscores';
      }
      return true;
    },
  });

  // Template selection
  const template = await select({
    message: 'Select a template:',
    choices: ['basic', 'full-stack', 'ml-training'],
    defaultValue: 'basic',
  });

  // Language selection
  const typescript = await confirm('Use TypeScript?', true);

  // Git initialization
  const gitInit = await confirm('Initialize git repository?', true);

  // Dependency installation
  const installDeps = await confirm('Install dependencies now?', true);

  return {
    name,
    template,
    typescript,
    gitInit,
    installDeps,
  };
}

/**
 * Display template details
 */
export function displayTemplateDetails(
  templateName: string,
  details: {
    description: string;
    features: string[];
    difficulty: string;
  }
): void {
  console.log(`\n📦 ${templateName}`);
  console.log('━'.repeat(50));
  console.log(`\n${details.description}\n`);
  console.log('Features:');
  details.features.forEach((feature) => {
    console.log(`  • ${feature}`);
  });
  console.log(`\nDifficulty: ${details.difficulty}\n`);
}

/**
 * Display progress indicator
 */
export class ProgressIndicator {
  private message: string;
  private interval: NodeJS.Timeout | null = null;
  private frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  private frameIndex = 0;

  constructor(message: string) {
    this.message = message;
  }

  start(): void {
    process.stdout.write(`${this.message} ${this.frames[0]}`);

    this.interval = setInterval(() => {
      this.frameIndex = (this.frameIndex + 1) % this.frames.length;
      process.stdout.write(`\r${this.message} ${this.frames[this.frameIndex]}`);
    }, 80);
  }

  stop(finalMessage?: string): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }

    if (finalMessage) {
      process.stdout.write(`\r${finalMessage}\n`);
    } else {
      process.stdout.write('\r' + ' '.repeat(this.message.length + 2) + '\r');
    }
  }
}
