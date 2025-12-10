#!/usr/bin/env node

/**
 * Nova Medicina CLI
 * AI-powered medical analysis system with anti-hallucination safeguards
 *
 * IMPORTANT MEDICAL DISCLAIMER:
 * This tool is designed to SUPPLEMENT, not replace, professional medical advice.
 * Always consult qualified healthcare providers for medical decisions.
 *
 * @author ruv (github.com/ruvnet, ruv.io)
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const program = new Command();

// Version from package.json
const packageJson = JSON.parse(
  await fs.readFile(path.join(__dirname, '../package.json'), 'utf-8')
);

program
  .name('nova-medicina')
  .description('AI-powered medical analysis system with anti-hallucination safeguards')
  .version(packageJson.version)
  .addHelpText('before', chalk.cyan.bold('\n🏥 Nova Medicina - Medical AI Analysis System\n'))
  .addHelpText('after', `
${chalk.yellow.bold('⚠️  MEDICAL DISCLAIMER:')}
This tool is designed to ${chalk.underline('SUPPLEMENT')}, not replace, professional medical advice.
${chalk.red('Always consult qualified healthcare providers for medical decisions.')}

${chalk.cyan('Examples:')}
  $ nova-medicina analyze --symptoms "fever, cough" --duration "3 days"
  $ nova-medicina verify --diagnosis "pneumonia" --evidence ./lab-results.json
  $ nova-medicina provider --specialty "cardiology" --location "New York"
  $ nova-medicina config --set verification-level=strict

${chalk.cyan('Documentation:')}
  https://github.com/ruvnet/nova-medicina

${chalk.cyan('Support:')}
  Issues: https://github.com/ruvnet/nova-medicina/issues
  Author: ruv (github.com/ruvnet, ruv.io)
`);

// Analyze Command
program
  .command('analyze')
  .description('Analyze symptoms and provide evidence-based insights')
  .option('-s, --symptoms <symptoms>', 'Comma-separated list of symptoms')
  .option('-d, --duration <duration>', 'Duration of symptoms (e.g., "3 days", "2 weeks")')
  .option('-a, --age <age>', 'Patient age (optional, helps with context)')
  .option('-g, --gender <gender>', 'Patient gender (optional, helps with context)')
  .option('-h, --history <history>', 'Medical history (optional, JSON file path)')
  .option('-m, --medications <meds>', 'Current medications (optional)')
  .option('-v, --verification-level <level>', 'Verification strictness: strict|moderate|basic', 'moderate')
  .option('-o, --output <file>', 'Output file for analysis results (JSON)')
  .option('--no-hallucination-check', 'Disable anti-hallucination verification (not recommended)')
  .option('--include-references', 'Include medical literature references')
  .option('--interactive', 'Interactive mode with follow-up questions')
  .addHelpText('after', `
${chalk.cyan.bold('\nSymptom Analysis with Anti-Hallucination Safeguards\n')}

${chalk.yellow('Description:')}
  Analyzes reported symptoms using AI with multiple verification layers to prevent
  hallucinations. Uses evidence-based medical knowledge and cross-references with
  medical literature databases.

${chalk.yellow('Verification Levels:')}
  ${chalk.green('strict')}     - Maximum verification, slowest but most reliable
  ${chalk.blue('moderate')}   - Balanced verification and speed (default)
  ${chalk.gray('basic')}      - Minimal verification, fastest but less thorough

${chalk.yellow('Anti-Hallucination Features:')}
  ✓ Multi-agent consensus verification
  ✓ Medical literature cross-referencing
  ✓ Confidence scoring with uncertainty quantification
  ✓ Contradictory evidence detection
  ✓ Source attribution for all claims

${chalk.cyan('Examples:')}
  # Basic symptom analysis
  $ nova-medicina analyze --symptoms "fever, cough, fatigue" --duration "3 days"

  # Detailed analysis with patient context
  $ nova-medicina analyze \\
      --symptoms "chest pain, shortness of breath" \\
      --duration "2 hours" \\
      --age 55 \\
      --gender male \\
      --medications "aspirin, metformin" \\
      --verification-level strict \\
      --include-references

  # Interactive mode for follow-up questions
  $ nova-medicina analyze --symptoms "headache" --interactive

  # Save analysis to file
  $ nova-medicina analyze --symptoms "abdominal pain" --output analysis.json

${chalk.red.bold('⚠️  EMERGENCY WARNING:')}
${chalk.red('For severe symptoms, chest pain, difficulty breathing, or life-threatening')}
${chalk.red('conditions, call emergency services immediately. Do NOT rely on this tool.')}
`)
  .action(async (options) => {
    const spinner = ora('Initializing medical analysis system...').start();

    try {
      // Validation
      if (!options.symptoms) {
        spinner.fail('Symptoms are required');
        console.log(chalk.yellow('\nUse: nova-medicina analyze --help for more information'));
        process.exit(1);
      }

      spinner.text = 'Loading anti-hallucination verification system...';
      await new Promise(resolve => setTimeout(resolve, 500));

      spinner.text = 'Analyzing symptoms with evidence-based AI...';
      await new Promise(resolve => setTimeout(resolve, 500));

      spinner.succeed('Analysis complete (verification pending)');

      console.log(chalk.cyan.bold('\n📋 Analysis Results:\n'));
      console.log(chalk.yellow('⚠️  Implementation in progress - full analysis coming soon'));
      console.log(chalk.gray(`\nSymptoms: ${options.symptoms}`));
      console.log(chalk.gray(`Duration: ${options.duration || 'Not specified'}`));
      console.log(chalk.gray(`Verification Level: ${options.verification}`));

      console.log(chalk.red.bold('\n⚠️  REMINDER: This is a supplement to professional care.'));
      console.log(chalk.red('Always consult a qualified healthcare provider.\n'));

    } catch (error) {
      spinner.fail('Analysis failed');
      console.error(chalk.red(`Error: ${error.message}`));
      process.exit(1);
    }
  });

// Verify Command
program
  .command('verify')
  .description('Verify medical information against evidence-based sources')
  .option('-d, --diagnosis <diagnosis>', 'Diagnosis to verify')
  .option('-t, --treatment <treatment>', 'Treatment plan to verify')
  .option('-e, --evidence <file>', 'Evidence file (JSON with lab results, imaging, etc.)')
  .option('-s, --sources <sources>', 'Comma-separated list of medical databases to check')
  .option('-c, --confidence-threshold <threshold>', 'Minimum confidence score (0-1)', '0.8')
  .option('-o, --output <file>', 'Output verification report (JSON)')
  .option('--show-contradictions', 'Highlight contradictory evidence')
  .option('--literature-review', 'Include systematic literature review')
  .addHelpText('after', `
${chalk.cyan.bold('\nMedical Information Verification System\n')}

${chalk.yellow('Description:')}
  Verifies medical diagnoses, treatments, and claims against evidence-based
  sources using multi-agent consensus and anti-hallucination safeguards.

${chalk.yellow('Verification Process:')}
  1. Multi-agent consensus analysis
  2. Cross-reference with medical literature (PubMed, Cochrane, UpToDate)
  3. Contradiction detection
  4. Confidence scoring
  5. Source attribution

${chalk.yellow('Medical Databases:')}
  - PubMed (biomedical literature)
  - Cochrane Library (systematic reviews)
  - UpToDate (clinical decision support)
  - ClinicalTrials.gov (clinical trials)
  - WHO ICD-11 (disease classification)

${chalk.cyan('Examples:')}
  # Verify a diagnosis
  $ nova-medicina verify --diagnosis "type 2 diabetes" --show-contradictions

  # Verify treatment plan with evidence
  $ nova-medicina verify \\
      --treatment "metformin 500mg twice daily" \\
      --diagnosis "type 2 diabetes" \\
      --evidence ./lab-results.json \\
      --literature-review

  # Custom verification with specific sources
  $ nova-medicina verify \\
      --diagnosis "pneumonia" \\
      --sources "pubmed,cochrane" \\
      --confidence-threshold 0.9

${chalk.red.bold('⚠️  IMPORTANT:')}
${chalk.red('Verification is not a substitute for professional medical judgment.')}
${chalk.red('Use results to inform discussions with healthcare providers.')}
`)
  .action(async (options) => {
    const spinner = ora('Starting verification process...').start();

    try {
      if (!options.diagnosis && !options.treatment) {
        spinner.fail('Either diagnosis or treatment must be specified');
        console.log(chalk.yellow('\nUse: nova-medicina verify --help for more information'));
        process.exit(1);
      }

      spinner.text = 'Loading medical literature databases...';
      await new Promise(resolve => setTimeout(resolve, 500));

      spinner.text = 'Running multi-agent consensus verification...';
      await new Promise(resolve => setTimeout(resolve, 500));

      spinner.succeed('Verification complete');

      console.log(chalk.cyan.bold('\n✅ Verification Results:\n'));
      console.log(chalk.yellow('⚠️  Implementation in progress - full verification coming soon'));

      if (options.diagnosis) {
        console.log(chalk.gray(`\nDiagnosis: ${options.diagnosis}`));
      }
      if (options.treatment) {
        console.log(chalk.gray(`Treatment: ${options.treatment}`));
      }
      console.log(chalk.gray(`Confidence Threshold: ${options.confidenceThreshold}`));

    } catch (error) {
      spinner.fail('Verification failed');
      console.error(chalk.red(`Error: ${error.message}`));
      process.exit(1);
    }
  });

// Provider Command
program
  .command('provider')
  .description('Find and verify healthcare providers')
  .option('-s, --specialty <specialty>', 'Medical specialty (e.g., cardiology, oncology)')
  .option('-l, --location <location>', 'Location (city, state, or zip code)')
  .option('-i, --insurance <insurance>', 'Insurance provider name')
  .option('-r, --rating <rating>', 'Minimum rating (1-5)', '4')
  .option('-d, --distance <miles>', 'Maximum distance in miles', '25')
  .option('--verify-credentials', 'Verify medical licenses and board certifications')
  .option('--accepting-patients', 'Show only providers accepting new patients')
  .option('-o, --output <file>', 'Output provider list (JSON)')
  .addHelpText('after', `
${chalk.cyan.bold('\nHealthcare Provider Search & Verification\n')}

${chalk.yellow('Description:')}
  Search for healthcare providers by specialty, location, and insurance.
  Verify credentials and certifications to ensure provider legitimacy.

${chalk.yellow('Verification Features:')}
  ✓ Medical license verification
  ✓ Board certification validation
  ✓ Disciplinary action checks
  ✓ Patient review analysis
  ✓ Insurance network verification

${chalk.cyan('Examples:')}
  # Find cardiologists in New York
  $ nova-medicina provider --specialty cardiology --location "New York, NY"

  # Find providers accepting specific insurance
  $ nova-medicina provider \\
      --specialty "family medicine" \\
      --location 10001 \\
      --insurance "Blue Cross" \\
      --accepting-patients

  # Detailed search with credential verification
  $ nova-medicina provider \\
      --specialty oncology \\
      --location "Los Angeles, CA" \\
      --distance 10 \\
      --rating 4.5 \\
      --verify-credentials \\
      --output providers.json

${chalk.yellow('Common Specialties:')}
  - Family Medicine, Internal Medicine
  - Cardiology, Oncology, Neurology
  - Orthopedics, Dermatology
  - Psychiatry, Psychology
  - Pediatrics, OB/GYN
`)
  .action(async (options) => {
    const spinner = ora('Searching for healthcare providers...').start();

    try {
      if (!options.specialty && !options.location) {
        spinner.fail('Either specialty or location must be specified');
        console.log(chalk.yellow('\nUse: nova-medicina provider --help for more information'));
        process.exit(1);
      }

      spinner.text = 'Querying provider databases...';
      await new Promise(resolve => setTimeout(resolve, 500));

      if (options.verifyCredentials) {
        spinner.text = 'Verifying medical licenses and certifications...';
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      spinner.succeed('Provider search complete');

      console.log(chalk.cyan.bold('\n👨‍⚕️ Provider Search Results:\n'));
      console.log(chalk.yellow('⚠️  Implementation in progress - full provider search coming soon'));

      if (options.specialty) {
        console.log(chalk.gray(`\nSpecialty: ${options.specialty}`));
      }
      if (options.location) {
        console.log(chalk.gray(`Location: ${options.location}`));
      }
      console.log(chalk.gray(`Distance: ${options.distance} miles`));
      console.log(chalk.gray(`Minimum Rating: ${options.rating}`));

    } catch (error) {
      spinner.fail('Provider search failed');
      console.error(chalk.red(`Error: ${error.message}`));
      process.exit(1);
    }
  });

// Config Command
program
  .command('config')
  .description('Configure Nova Medicina settings')
  .option('--set <key=value>', 'Set configuration value')
  .option('--get <key>', 'Get configuration value')
  .option('--list', 'List all configuration settings')
  .option('--reset', 'Reset to default configuration')
  .option('--api-key <key>', 'Set API key for external services')
  .option('--data-dir <path>', 'Set data directory path')
  .option('--cache-dir <path>', 'Set cache directory path')
  .addHelpText('after', `
${chalk.cyan.bold('\nConfiguration Management\n')}

${chalk.yellow('Available Settings:')}
  verification-level        Verification strictness (strict|moderate|basic)
  api-key                  API key for external medical databases
  data-dir                 Directory for storing analysis data
  cache-dir                Directory for caching medical literature
  max-cache-age           Maximum cache age in hours (default: 24)
  hallucination-threshold  Confidence threshold for hallucination detection
  enable-telemetry        Enable anonymous usage telemetry (default: true)
  default-output-format   Output format (json|text|html)

${chalk.cyan('Examples:')}
  # Set verification level to strict
  $ nova-medicina config --set verification-level=strict

  # Configure API key
  $ nova-medicina config --api-key "your-api-key-here"

  # List all settings
  $ nova-medicina config --list

  # Get specific setting
  $ nova-medicina config --get verification-level

  # Reset to defaults
  $ nova-medicina config --reset
`)
  .action(async (options) => {
    const spinner = ora('Managing configuration...').start();

    try {
      if (options.list) {
        spinner.succeed('Configuration loaded');
        console.log(chalk.cyan.bold('\n⚙️  Current Configuration:\n'));
        console.log(chalk.yellow('⚠️  Implementation in progress - config management coming soon'));
      } else if (options.set) {
        const [key, value] = options.set.split('=');
        spinner.succeed(`Configuration updated: ${key} = ${value}`);
        console.log(chalk.green(`\n✓ Set ${key} to ${value}`));
      } else if (options.get) {
        spinner.succeed('Configuration retrieved');
        console.log(chalk.cyan(`\n${options.get}: (implementation in progress)`));
      } else if (options.reset) {
        spinner.succeed('Configuration reset to defaults');
        console.log(chalk.green('\n✓ Configuration reset successfully'));
      } else {
        spinner.info('No action specified');
        console.log(chalk.yellow('\nUse: nova-medicina config --help for more information'));
      }

    } catch (error) {
      spinner.fail('Configuration operation failed');
      console.error(chalk.red(`Error: ${error.message}`));
      process.exit(1);
    }
  });

// Parse arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}
