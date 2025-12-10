#!/usr/bin/env node
/**
 * DAA SDK CLI Tool
 *
 * Provides command-line utilities for:
 * - Project initialization with templates
 * - Development server
 * - Testing and benchmarking
 * - Deployment
 *
 * @example
 * ```bash
 * npx daa-sdk init my-agent --template full-stack
 * npx daa-sdk dev --hot-reload
 * npx daa-sdk test --runtime native,wasm
 * npx daa-sdk benchmark --output report.html
 * ```
 */

import { Command } from 'commander';
import chalk from 'chalk';
import { detectPlatform, getPlatformInfo, getAvailableBindings } from '../src/platform';

const program = new Command();

program
  .name('daa-sdk')
  .description('DAA SDK - Command-line tools for Distributed Agentic Architecture')
  .version('0.1.0');

/**
 * Initialize a new DAA project
 */
program
  .command('init [name]')
  .description('Initialize a new DAA project')
  .option('-t, --template <type>', 'Project template (basic|full-stack|ml-training)', 'basic')
  .option('--no-install', 'Skip dependency installation')
  .option('--no-git', 'Skip git initialization')
  .option('--typescript', 'Use TypeScript (default)', true)
  .option('--javascript', 'Use JavaScript instead of TypeScript')
  .action(async (name: string | undefined, options: any) => {
    const {
      scaffoldProject,
      validateProjectName,
      generateProjectSummary,
      TEMPLATES,
      getTemplateInfo,
    } = await import('./templates');
    const {
      interactiveSetup,
      confirm,
      displayTemplateDetails,
      ProgressIndicator,
    } = await import('./prompts');

    try {
      // Interactive mode if no name provided
      if (!name) {
        const setup = await interactiveSetup();
        name = setup.name;
        options.template = setup.template;
        options.install = setup.installDeps;
        options.git = setup.gitInit;
      }

      // Validate project name
      const validation = validateProjectName(name);
      if (!validation.valid) {
        console.log(chalk.red(`\n❌ ${validation.error}\n`));
        process.exit(1);
      }

      // Validate template
      const templateInfo = getTemplateInfo(options.template);
      if (!templateInfo) {
        console.log(chalk.red(`\n❌ Unknown template: ${options.template}\n`));
        console.log(chalk.gray('Available templates:'));
        Object.keys(TEMPLATES).forEach((t) => {
          console.log(chalk.gray(`  • ${t}`));
        });
        console.log();
        process.exit(1);
      }

      console.log(chalk.blue('\n🚀 Creating DAA Project\n'));

      // Display template details
      displayTemplateDetails(templateInfo.name, templateInfo);

      // Confirm creation
      const shouldCreate = await confirm('Create project with these settings?', true);
      if (!shouldCreate) {
        console.log(chalk.yellow('\n⚠️  Project creation cancelled\n'));
        process.exit(0);
      }

      // Create project
      const progress = new ProgressIndicator('📦 Creating project');
      progress.start();

      const scaffoldOptions = {
        name,
        template: options.template,
        typescript: !options.javascript,
        installDeps: options.install !== false,
        gitInit: options.git !== false,
      };

      await scaffoldProject(scaffoldOptions);

      progress.stop(chalk.green('✅ Project created successfully!'));

      // Display summary
      console.log(generateProjectSummary(scaffoldOptions));
    } catch (error: any) {
      console.log(chalk.red(`\n❌ Error: ${error.message}\n`));
      process.exit(1);
    }
  });

/**
 * Show platform information
 */
program
  .command('info')
  .description('Show platform and binding information')
  .action(async () => {
    console.log(chalk.blue('\n📊 DAA SDK Platform Information\n'));

    const platform = detectPlatform();
    const info = getPlatformInfo();
    const bindings = await getAvailableBindings();

    console.log(chalk.bold('Platform:'), info.platform);
    console.log(chalk.bold('Runtime:'), info.runtime);
    console.log(chalk.bold('Performance:'), info.performance);
    console.log(chalk.bold('Relative Speed:'), `${info.relativeSpeed * 100}%`);
    console.log(chalk.bold('Threading:'), info.threadingSupport ? '✅ Supported' : '❌ Not supported');

    console.log(chalk.bold('\nFeatures:'));
    info.features.forEach((feature) => {
      console.log(chalk.gray(`  • ${feature}`));
    });

    console.log(chalk.bold('\nAvailable Bindings:'));
    bindings.available.forEach((binding) => {
      console.log(chalk.green(`  ✅ ${binding}`));
    });

    if (bindings.unavailable.length > 0) {
      console.log(chalk.bold('\nUnavailable Bindings:'));
      bindings.unavailable.forEach((binding) => {
        console.log(chalk.yellow(`  ⚠️  ${binding} (not yet implemented)`));
      });
    }

    console.log();
  });

/**
 * Run development server
 */
program
  .command('dev')
  .description('Start development server')
  .option('-p, --port <port>', 'Server port', '3000')
  .option('--hot-reload', 'Enable hot module reloading')
  .action((options: any) => {
    console.log(chalk.blue(`\n🔧 Starting development server on port ${options.port}\n`));
    console.log(chalk.yellow('⚠️  Development server not yet implemented\n'));
  });

/**
 * Run tests
 */
program
  .command('test')
  .description('Run test suite')
  .option('--runtime <runtimes>', 'Test specific runtimes (native,wasm)', 'native,wasm')
  .option('--coverage', 'Generate coverage report')
  .action((options: any) => {
    console.log(chalk.blue('\n🧪 Running tests\n'));
    console.log(chalk.gray('Runtimes:'), options.runtime);
    console.log(chalk.yellow('⚠️  Test runner not yet implemented\n'));
  });

/**
 * Run benchmarks
 */
program
  .command('benchmark')
  .description('Run performance benchmarks')
  .option('--compare <runtimes>', 'Compare runtimes (native,wasm)', 'native,wasm')
  .option('--output <format>', 'Output format (json|html|text)', 'text')
  .option('--iterations <n>', 'Number of iterations', '1000')
  .action((options: any) => {
    console.log(chalk.blue('\n⚡ Running benchmarks\n'));
    console.log(chalk.gray('Comparing:'), options.compare);
    console.log(chalk.gray('Iterations:'), options.iterations);
    console.log(chalk.yellow('⚠️  Benchmark suite not yet implemented\n'));
  });

/**
 * Deploy to production
 */
program
  .command('deploy')
  .description('Deploy to production')
  .option('--target <env>', 'Deployment target (cloud|edge|local)', 'cloud')
  .option('--optimize', 'Enable production optimizations')
  .action((options: any) => {
    console.log(chalk.blue(`\n🚀 Deploying to ${options.target}\n`));
    console.log(chalk.yellow('⚠️  Deployment not yet implemented\n'));
  });

/**
 * List available templates
 */
program
  .command('templates')
  .description('List available project templates')
  .action(async () => {
    const { listTemplates, getTemplateInfo } = await import('./templates');
    const { displayTemplateDetails } = await import('./prompts');

    console.log(chalk.blue('\n📦 Available Templates\n'));

    const templates = listTemplates();
    templates.forEach((template) => {
      displayTemplateDetails(template.name, template);
    });

    console.log(chalk.gray('💡 Create a new project: npx daa-sdk init my-project\n'));
  });

/**
 * Show examples
 */
program
  .command('examples')
  .description('Show usage examples')
  .option('-t, --template <type>', 'Show examples for specific template')
  .action((options: any) => {
    console.log(chalk.blue('\n📚 DAA SDK Examples\n'));

    if (options.template) {
      // Show template-specific examples
      if (options.template === 'basic') {
        console.log(chalk.bold('Basic Template Examples:'));
        console.log(chalk.gray(`
  // ML-KEM Key Encapsulation
  const mlkem = daa.crypto.mlkem();
  const keypair = mlkem.generateKeypair();
  const { ciphertext, sharedSecret } = mlkem.encapsulate(keypair.publicKey);

  // Digital Signatures
  const mldsa = daa.crypto.mldsa();
  const signature = mldsa.sign(secretKey, message);
  const isValid = mldsa.verify(publicKey, message, signature);
        `));
      } else if (options.template === 'full-stack') {
        console.log(chalk.bold('Full-Stack Template Examples:'));
        console.log(chalk.gray(`
  // Start MRAP Orchestrator
  await daa.orchestrator.start();

  // Create Workflow
  const workflow = await daa.orchestrator.createWorkflow({
    steps: [/* workflow definition */],
  });

  // Token Economy
  await daa.economy.transfer('agent-1', 'agent-2', 100);
        `));
      } else if (options.template === 'ml-training') {
        console.log(chalk.bold('ML Training Template Examples:'));
        console.log(chalk.gray(`
  // Start Federated Training
  const session = await daa.prime.startTraining({
    model: 'gpt-mini',
    nodes: 10,
    privacy: { differentialPrivacy: true, epsilon: 1.0 },
  });

  // Monitor Progress
  const progress = await daa.prime.getProgress(session.id);
        `));
      }
    } else {
      // Show general examples
      console.log(chalk.bold('1. Basic Usage:'));
      console.log(chalk.gray(`
  import { DAA } from 'daa-sdk';

  const daa = new DAA();
  await daa.init();

  // Use quantum-resistant crypto
  const mlkem = daa.crypto.mlkem();
  const keypair = mlkem.generateKeypair();
      `));

      console.log(chalk.bold('2. Orchestrator:'));
      console.log(chalk.gray(`
  import { DAA } from 'daa-sdk';

  const daa = new DAA({ orchestrator: { enableMRAP: true } });
  await daa.init();

  // Start MRAP autonomy loop
  await daa.orchestrator.start();

  // Monitor system state
  const state = await daa.orchestrator.monitor();
      `));

      console.log(chalk.bold('3. Federated Learning:'));
      console.log(chalk.gray(`
  import { DAA } from 'daa-sdk';

  const daa = new DAA({ prime: { enableTraining: true } });
  await daa.init();

  // Start training
  const session = await daa.prime.startTraining({
    model: 'gpt-mini',
    nodes: 10,
  });
      `));

      console.log(chalk.gray('\n💡 See template-specific examples: npx daa-sdk examples --template <name>\n'));
    }
  });

// Parse command line arguments
program.parse();

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}
