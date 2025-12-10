/**
 * Nova Medicina - Comprehensive CLI Help System
 *
 * Provides detailed help, examples, warnings, and interactive guidance
 * for the nova-medicina emergency medical analysis CLI.
 */

import chalk from 'chalk';
import boxen from 'boxen';

// ASCII Art Logo
const LOGO = `
███╗   ██╗ ██████╗ ██╗   ██╗ █████╗     ███╗   ███╗███████╗██████╗ ██╗ ██████╗██╗███╗   ██╗ █████╗
████╗  ██║██╔═══██╗██║   ██║██╔══██╗    ████╗ ████║██╔════╝██╔══██╗██║██╔════╝██║████╗  ██║██╔══██╗
██╔██╗ ██║██║   ██║██║   ██║███████║    ██╔████╔██║█████╗  ██║  ██║██║██║     ██║██╔██╗ ██║███████║
██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║    ██║╚██╔╝██║██╔══╝  ██║  ██║██║██║     ██║██║╚██╗██║██╔══██║
██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║    ██║ ╚═╝ ██║███████╗██████╔╝██║╚██████╗██║██║ ╚████║██║  ██║
╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝    ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝

                    🚨 Emergency Medical Analysis & Triage Assistant 🚨
`;

// Safety Warning Box
const SAFETY_WARNING = boxen(
  chalk.red.bold('⚠️  CRITICAL SAFETY WARNING ⚠️\n\n') +
  chalk.yellow(
    'This tool is NOT a substitute for professional medical care.\n' +
    'In a medical emergency:\n\n' +
    '  • Call emergency services immediately (911 in US)\n' +
    '  • Seek immediate medical attention\n' +
    '  • Follow provider recommendations\n\n' +
    'This tool provides analysis for INFORMATIONAL purposes only.\n' +
    'Always consult qualified healthcare professionals for medical decisions.'
  ),
  {
    padding: 1,
    margin: 1,
    borderStyle: 'double',
    borderColor: 'red',
    backgroundColor: '#330000'
  }
);

interface HelpSection {
  title: string;
  content: string;
  examples?: string[];
  warnings?: string[];
}

/**
 * Main help display
 */
export function showMainHelp(): void {
  console.log(chalk.cyan(LOGO));
  console.log(SAFETY_WARNING);

  console.log(chalk.bold.white('\n📋 OVERVIEW\n'));
  console.log(chalk.gray(
    'Nova Medicina is an AI-powered emergency medical analysis tool that helps\n' +
    'analyze symptoms, assess severity, and coordinate with healthcare providers.\n'
  ));

  console.log(chalk.bold.white('\n🎯 QUICK START\n'));
  console.log(chalk.cyan('  1.') + ' Configure providers:  ' + chalk.green('nova-medicina config'));
  console.log(chalk.cyan('  2.') + ' Analyze symptoms:     ' + chalk.green('nova-medicina analyze "chest pain, shortness of breath"'));
  console.log(chalk.cyan('  3.') + ' Verify with AI:       ' + chalk.green('nova-medicina verify <analysis-id>'));
  console.log(chalk.cyan('  4.') + ' Notify providers:     ' + chalk.green('nova-medicina provider notify <analysis-id>'));

  console.log(chalk.bold.white('\n📚 COMMANDS\n'));

  const commands = [
    {
      name: 'analyze',
      desc: 'Analyze symptoms and assess medical urgency',
      usage: 'nova-medicina analyze [options] "<symptoms>"'
    },
    {
      name: 'verify',
      desc: 'Verify analysis with multiple AI models',
      usage: 'nova-medicina verify <analysis-id>'
    },
    {
      name: 'provider',
      desc: 'Manage and notify healthcare providers',
      usage: 'nova-medicina provider <command> [options]'
    },
    {
      name: 'config',
      desc: 'Configure CLI settings and API keys',
      usage: 'nova-medicina config [options]'
    },
    {
      name: 'history',
      desc: 'View analysis history',
      usage: 'nova-medicina history [options]'
    },
    {
      name: 'export',
      desc: 'Export analysis data',
      usage: 'nova-medicina export <analysis-id> [format]'
    }
  ];

  commands.forEach(cmd => {
    console.log(chalk.bold.green(`  ${cmd.name.padEnd(12)}`), chalk.gray(cmd.desc));
    console.log(chalk.dim(`               ${cmd.usage}\n`));
  });

  console.log(chalk.bold.white('📖 DETAILED HELP\n'));
  console.log(chalk.gray('  For detailed help on any command, use:\n'));
  console.log(chalk.cyan('    nova-medicina <command> --help'));
  console.log(chalk.cyan('    nova-medicina help <command>'));
  console.log(chalk.cyan('    nova-medicina --tutorial') + chalk.gray('  (Interactive tutorial mode)\n'));

  console.log(chalk.bold.white('🔗 RESOURCES\n'));
  console.log(chalk.gray('  Documentation: ') + chalk.blue('https://nova-medicina.io/docs'));
  console.log(chalk.gray('  GitHub:        ') + chalk.blue('https://github.com/nova-medicina/cli'));
  console.log(chalk.gray('  Support:       ') + chalk.blue('support@nova-medicina.io'));
  console.log(chalk.gray('  Version:       ') + chalk.yellow('1.0.0\n'));
}

/**
 * Analyze command help
 */
export function showAnalyzeHelp(): void {
  console.log(chalk.bold.cyan('\n🔍 ANALYZE COMMAND\n'));
  console.log(chalk.gray('Analyze symptoms and assess medical urgency using AI-powered triage.\n'));

  console.log(chalk.bold.white('USAGE:\n'));
  console.log(chalk.green('  nova-medicina analyze [options] "<symptoms>"\n'));

  console.log(chalk.bold.white('OPTIONS:\n'));
  const options = [
    ['--severity <level>', 'Filter by severity (mild, moderate, severe, critical)'],
    ['--age <years>', 'Patient age (affects risk assessment)'],
    ['--conditions <list>', 'Pre-existing conditions (comma-separated)'],
    ['--medications <list>', 'Current medications (comma-separated)'],
    ['--allergies <list>', 'Known allergies (comma-separated)'],
    ['--duration <time>', 'Symptom duration (e.g., "2 hours", "3 days")'],
    ['--photo <path>', 'Include photo of affected area'],
    ['--voice <path>', 'Include voice recording of symptoms'],
    ['--emergency', 'Flag as potential emergency (higher priority)'],
    ['--lang <code>', 'Language code (default: en)'],
    ['--json', 'Output in JSON format'],
    ['--save <path>', 'Save analysis to file']
  ];

  options.forEach(([flag, desc]) => {
    console.log(chalk.cyan(`  ${flag.padEnd(25)}`), chalk.gray(desc));
  });

  console.log(chalk.bold.white('\n💡 EXAMPLES:\n'));

  const examples = [
    {
      desc: 'Basic symptom analysis',
      cmd: 'nova-medicina analyze "headache and fever for 2 days"'
    },
    {
      desc: 'Emergency with patient details',
      cmd: 'nova-medicina analyze --emergency --age 45 "severe chest pain, radiating to left arm"'
    },
    {
      desc: 'Analysis with medical history',
      cmd: 'nova-medicina analyze --age 65 --conditions "diabetes,hypertension" "dizziness and blurred vision"'
    },
    {
      desc: 'Include photo evidence',
      cmd: 'nova-medicina analyze --photo ./rash.jpg "red itchy rash on arms"'
    },
    {
      desc: 'Multi-symptom analysis',
      cmd: 'nova-medicina analyze "fever (102°F), cough, difficulty breathing, fatigue" --duration "3 days"'
    }
  ];

  examples.forEach((ex, i) => {
    console.log(chalk.yellow(`  ${i + 1}. ${ex.desc}`));
    console.log(chalk.dim(`     ${ex.cmd}\n`));
  });

  console.log(chalk.bold.white('⚕️  SYMPTOM CATEGORIES:\n'));
  console.log(chalk.gray('  Common symptoms the analyzer can assess:\n'));

  const categories = {
    'Cardiovascular': ['chest pain', 'heart palpitations', 'shortness of breath'],
    'Neurological': ['headache', 'dizziness', 'confusion', 'numbness'],
    'Respiratory': ['cough', 'wheezing', 'difficulty breathing'],
    'Digestive': ['abdominal pain', 'nausea', 'vomiting', 'diarrhea'],
    'Musculoskeletal': ['joint pain', 'back pain', 'muscle weakness'],
    'Dermatological': ['rash', 'swelling', 'discoloration'],
    'General': ['fever', 'fatigue', 'weight loss']
  };

  Object.entries(categories).forEach(([category, symptoms]) => {
    console.log(chalk.cyan(`  ${category}:`), chalk.gray(symptoms.join(', ')));
  });

  console.log(chalk.bold.white('\n🚨 EMERGENCY INDICATORS:\n'));
  console.log(chalk.red('  The following symptoms require IMMEDIATE medical attention:\n'));

  const emergencies = [
    'Chest pain or pressure',
    'Difficulty breathing or shortness of breath',
    'Severe bleeding that won\'t stop',
    'Sudden severe headache',
    'Loss of consciousness',
    'Severe allergic reaction',
    'Stroke symptoms (facial drooping, arm weakness, speech difficulty)',
    'Suspected poisoning',
    'Severe burns',
    'Suicidal thoughts or behavior'
  ];

  emergencies.forEach(symptom => {
    console.log(chalk.red('  •'), chalk.yellow(symptom));
  });

  console.log(chalk.bold.red('\n  ⚠️  IF EXPERIENCING ANY ABOVE: CALL 911 IMMEDIATELY ⚠️\n'));

  console.log(chalk.bold.white('📊 OUTPUT:\n'));
  console.log(chalk.gray('  Analysis includes:\n'));
  console.log(chalk.gray('    • Severity assessment (mild/moderate/severe/critical)'));
  console.log(chalk.gray('    • Possible conditions'));
  console.log(chalk.gray('    • Recommended actions'));
  console.log(chalk.gray('    • Urgency timeline'));
  console.log(chalk.gray('    • Warning signs to monitor'));
  console.log(chalk.gray('    • Provider notification recommendations\n'));
}

/**
 * Verify command help
 */
export function showVerifyHelp(): void {
  console.log(chalk.bold.cyan('\n✓ VERIFY COMMAND\n'));
  console.log(chalk.gray('Verify analysis using multiple AI models for consensus and confidence scoring.\n'));

  console.log(chalk.bold.white('USAGE:\n'));
  console.log(chalk.green('  nova-medicina verify <analysis-id> [options]\n'));

  console.log(chalk.bold.white('OPTIONS:\n'));
  const options = [
    ['--models <list>', 'Specific models to use (comma-separated)'],
    ['--min-confidence <n>', 'Minimum confidence threshold (0-100, default: 75)'],
    ['--consensus', 'Require model consensus (majority agreement)'],
    ['--explain', 'Show detailed explanation of confidence scoring'],
    ['--json', 'Output in JSON format']
  ];

  options.forEach(([flag, desc]) => {
    console.log(chalk.cyan(`  ${flag.padEnd(25)}`), chalk.gray(desc));
  });

  console.log(chalk.bold.white('\n🤖 VERIFICATION MODELS:\n'));
  console.log(chalk.gray('  Nova Medicina uses multiple AI models for verification:\n'));

  const models = [
    { name: 'GPT-4o', specialty: 'General medical analysis' },
    { name: 'Claude-4-Sonnet', specialty: 'Detailed reasoning and safety' },
    { name: 'Gemini-2.0', specialty: 'Pattern recognition' },
    { name: 'Med-PaLM', specialty: 'Medical knowledge specialization' }
  ];

  models.forEach(model => {
    console.log(chalk.cyan(`  ${model.name.padEnd(20)}`), chalk.gray(model.specialty));
  });

  console.log(chalk.bold.white('\n📊 CONFIDENCE SCORING:\n'));
  console.log(chalk.gray('  Confidence scores are calculated based on:\n'));
  console.log(chalk.gray('    • Model agreement (consensus across models)'));
  console.log(chalk.gray('    • Individual model confidence'));
  console.log(chalk.gray('    • Evidence strength from symptoms'));
  console.log(chalk.gray('    • Medical knowledge base alignment'));
  console.log(chalk.gray('    • Historical accuracy patterns\n'));

  console.log(chalk.bold.white('  Confidence Levels:\n'));
  console.log(chalk.green('    90-100%  '), chalk.gray('Very High - Strong model consensus'));
  console.log(chalk.cyan('    75-89%   '), chalk.gray('High - Good agreement'));
  console.log(chalk.yellow('    60-74%   '), chalk.gray('Moderate - Some disagreement'));
  console.log(chalk.red('    Below 60%'), chalk.gray('Low - Significant disagreement (recommend human review)\n'));

  console.log(chalk.bold.white('💡 EXAMPLES:\n'));

  const examples = [
    {
      desc: 'Verify analysis with default settings',
      cmd: 'nova-medicina verify abc123def456'
    },
    {
      desc: 'Verify with specific models',
      cmd: 'nova-medicina verify abc123def456 --models "gpt-4o,claude-4-sonnet"'
    },
    {
      desc: 'Require high confidence',
      cmd: 'nova-medicina verify abc123def456 --min-confidence 85 --consensus'
    },
    {
      desc: 'Detailed verification explanation',
      cmd: 'nova-medicina verify abc123def456 --explain'
    }
  ];

  examples.forEach((ex, i) => {
    console.log(chalk.yellow(`  ${i + 1}. ${ex.desc}`));
    console.log(chalk.dim(`     ${ex.cmd}\n`));
  });

  console.log(chalk.bold.white('🔍 VERIFICATION PROCESS:\n'));
  console.log(chalk.gray('  1. Original analysis is sent to multiple AI models'));
  console.log(chalk.gray('  2. Each model independently analyzes symptoms'));
  console.log(chalk.gray('  3. Results are compared for consensus'));
  console.log(chalk.gray('  4. Confidence score is calculated'));
  console.log(chalk.gray('  5. Discrepancies are flagged for review'));
  console.log(chalk.gray('  6. Final verified analysis is generated\n'));
}

/**
 * Provider command help
 */
export function showProviderHelp(): void {
  console.log(chalk.bold.cyan('\n👨‍⚕️ PROVIDER COMMAND\n'));
  console.log(chalk.gray('Manage healthcare provider contacts and send notifications.\n'));

  console.log(chalk.bold.white('USAGE:\n'));
  console.log(chalk.green('  nova-medicina provider <command> [options]\n'));

  console.log(chalk.bold.white('COMMANDS:\n'));

  const commands = [
    {
      name: 'add',
      desc: 'Add a healthcare provider',
      usage: 'nova-medicina provider add --name "Dr. Smith" --email "dr.smith@clinic.com"'
    },
    {
      name: 'list',
      desc: 'List all configured providers',
      usage: 'nova-medicina provider list'
    },
    {
      name: 'remove',
      desc: 'Remove a provider',
      usage: 'nova-medicina provider remove <provider-id>'
    },
    {
      name: 'notify',
      desc: 'Send analysis to provider',
      usage: 'nova-medicina provider notify <analysis-id> [options]'
    },
    {
      name: 'status',
      desc: 'Check notification status',
      usage: 'nova-medicina provider status <notification-id>'
    }
  ];

  commands.forEach(cmd => {
    console.log(chalk.bold.green(`  ${cmd.name.padEnd(12)}`), chalk.gray(cmd.desc));
    console.log(chalk.dim(`               ${cmd.usage}\n`));
  });

  console.log(chalk.bold.white('ADD PROVIDER OPTIONS:\n'));
  const addOptions = [
    ['--name <name>', 'Provider name (required)'],
    ['--email <email>', 'Email address'],
    ['--phone <number>', 'Phone number'],
    ['--fax <number>', 'Fax number'],
    ['--specialty <type>', 'Medical specialty'],
    ['--clinic <name>', 'Clinic/hospital name'],
    ['--primary', 'Set as primary care provider'],
    ['--emergency', 'Emergency contact']
  ];

  addOptions.forEach(([flag, desc]) => {
    console.log(chalk.cyan(`  ${flag.padEnd(25)}`), chalk.gray(desc));
  });

  console.log(chalk.bold.white('\n📧 NOTIFICATION OPTIONS:\n'));
  const notifyOptions = [
    ['--provider <id>', 'Specific provider (default: all)'],
    ['--method <type>', 'email, sms, fax, or portal (default: email)'],
    ['--urgent', 'Flag as urgent notification'],
    ['--include-images', 'Include photos in notification'],
    ['--message <text>', 'Add personal message'],
    ['--cc <emails>', 'CC additional recipients'],
    ['--request-callback', 'Request callback from provider']
  ];

  notifyOptions.forEach(([flag, desc]) => {
    console.log(chalk.cyan(`  ${flag.padEnd(25)}`), chalk.gray(desc));
  });

  console.log(chalk.bold.white('\n💡 EXAMPLES:\n'));

  const examples = [
    {
      desc: 'Add primary care provider',
      cmd: 'nova-medicina provider add --name "Dr. Jane Smith" --email "jsmith@clinic.com" --phone "555-0100" --primary'
    },
    {
      desc: 'Add emergency contact',
      cmd: 'nova-medicina provider add --name "City Hospital ER" --phone "555-0911" --emergency'
    },
    {
      desc: 'Notify all providers',
      cmd: 'nova-medicina provider notify abc123def456'
    },
    {
      desc: 'Urgent notification to specific provider',
      cmd: 'nova-medicina provider notify abc123def456 --provider dr-smith --urgent --request-callback'
    },
    {
      desc: 'Send via multiple methods',
      cmd: 'nova-medicina provider notify abc123def456 --method email,sms --include-images'
    }
  ];

  examples.forEach((ex, i) => {
    console.log(chalk.yellow(`  ${i + 1}. ${ex.desc}`));
    console.log(chalk.dim(`     ${ex.cmd}\n`));
  });

  console.log(chalk.bold.white('🔐 PRIVACY & SECURITY:\n'));
  console.log(chalk.gray('  • Provider data is encrypted at rest'));
  console.log(chalk.gray('  • HIPAA-compliant transmission'));
  console.log(chalk.gray('  • Audit trail for all notifications'));
  console.log(chalk.gray('  • Secure portal integration available'));
  console.log(chalk.gray('  • Patient consent required for sharing\n'));

  console.log(chalk.bold.white('📋 NOTIFICATION CONTENT:\n'));
  console.log(chalk.gray('  Notifications include:\n'));
  console.log(chalk.gray('    • Patient demographics (if configured)'));
  console.log(chalk.gray('    • Symptom analysis summary'));
  console.log(chalk.gray('    • Severity assessment'));
  console.log(chalk.gray('    • AI confidence scores'));
  console.log(chalk.gray('    • Recommended actions'));
  console.log(chalk.gray('    • Supporting images/data (if included)'));
  console.log(chalk.gray('    • Contact information for follow-up\n'));
}

/**
 * Config command help
 */
export function showConfigHelp(): void {
  console.log(chalk.bold.cyan('\n⚙️  CONFIG COMMAND\n'));
  console.log(chalk.gray('Configure Nova Medicina CLI settings, API keys, and preferences.\n'));

  console.log(chalk.bold.white('USAGE:\n'));
  console.log(chalk.green('  nova-medicina config [options]\n'));
  console.log(chalk.green('  nova-medicina config set <key> <value>'));
  console.log(chalk.green('  nova-medicina config get <key>'));
  console.log(chalk.green('  nova-medicina config list\n'));

  console.log(chalk.bold.white('OPTIONS:\n'));
  const options = [
    ['--api-key <key>', 'Set Nova Medicina API key'],
    ['--openai-key <key>', 'Set OpenAI API key'],
    ['--anthropic-key <key>', 'Set Anthropic API key'],
    ['--google-key <key>', 'Set Google AI API key'],
    ['--region <code>', 'Set region (us, eu, asia)'],
    ['--language <code>', 'Default language (en, es, fr, de, etc.)'],
    ['--output <format>', 'Default output format (text, json)'],
    ['--auto-verify', 'Enable automatic verification'],
    ['--save-history', 'Save analysis history'],
    ['--encrypt-data', 'Enable data encryption']
  ];

  options.forEach(([flag, desc]) => {
    console.log(chalk.cyan(`  ${flag.padEnd(25)}`), chalk.gray(desc));
  });

  console.log(chalk.bold.white('\n🔑 SETUP GUIDE:\n'));
  console.log(chalk.yellow('  Step 1: Get API Keys\n'));
  console.log(chalk.gray('    Visit https://nova-medicina.io/api to get your API key'));
  console.log(chalk.gray('    (Optional) Get keys for AI verification:\n'));
  console.log(chalk.gray('      • OpenAI: https://platform.openai.com/api-keys'));
  console.log(chalk.gray('      • Anthropic: https://console.anthropic.com/settings/keys'));
  console.log(chalk.gray('      • Google AI: https://makersuite.google.com/app/apikey\n'));

  console.log(chalk.yellow('  Step 2: Configure CLI\n'));
  console.log(chalk.dim('    nova-medicina config --api-key YOUR_KEY_HERE\n'));

  console.log(chalk.yellow('  Step 3: Set Preferences\n'));
  console.log(chalk.dim('    nova-medicina config --region us --language en --auto-verify\n'));

  console.log(chalk.yellow('  Step 4: Add Providers\n'));
  console.log(chalk.dim('    nova-medicina provider add --name "Dr. Smith" --email "dr.smith@clinic.com"\n'));

  console.log(chalk.yellow('  Step 5: Test Configuration\n'));
  console.log(chalk.dim('    nova-medicina analyze "test headache" --json\n'));

  console.log(chalk.bold.white('📂 CONFIGURATION LOCATIONS:\n'));
  console.log(chalk.gray('  Config file:    ') + chalk.blue('~/.nova-medicina/config.json'));
  console.log(chalk.gray('  Provider data:  ') + chalk.blue('~/.nova-medicina/providers.json'));
  console.log(chalk.gray('  History:        ') + chalk.blue('~/.nova-medicina/history/'));
  console.log(chalk.gray('  Cache:          ') + chalk.blue('~/.nova-medicina/cache/'));
  console.log(chalk.gray('  Logs:           ') + chalk.blue('~/.nova-medicina/logs/\n'));

  console.log(chalk.bold.white('💡 CONFIGURATION EXAMPLES:\n'));

  const examples = [
    {
      desc: 'Quick setup',
      cmd: 'nova-medicina config --api-key "nm_abc123" --auto-verify'
    },
    {
      desc: 'Set specific value',
      cmd: 'nova-medicina config set output json'
    },
    {
      desc: 'View configuration',
      cmd: 'nova-medicina config list'
    },
    {
      desc: 'Enable all features',
      cmd: 'nova-medicina config --auto-verify --save-history --encrypt-data'
    }
  ];

  examples.forEach((ex, i) => {
    console.log(chalk.yellow(`  ${i + 1}. ${ex.desc}`));
    console.log(chalk.dim(`     ${ex.cmd}\n`));
  });

  console.log(chalk.bold.white('🔐 SECURITY BEST PRACTICES:\n'));
  console.log(chalk.gray('  • Never share your API keys'));
  console.log(chalk.gray('  • Use environment variables for keys in CI/CD'));
  console.log(chalk.gray('  • Enable data encryption for sensitive information'));
  console.log(chalk.gray('  • Regularly rotate API keys'));
  console.log(chalk.gray('  • Review access logs periodically\n'));

  console.log(chalk.bold.white('🌍 ENVIRONMENT VARIABLES:\n'));
  console.log(chalk.gray('  You can also set configuration via environment variables:\n'));
  console.log(chalk.cyan('    NOVA_MEDICINA_API_KEY    '), chalk.gray('Main API key'));
  console.log(chalk.cyan('    NOVA_MEDICINA_REGION     '), chalk.gray('Region code'));
  console.log(chalk.cyan('    NOVA_MEDICINA_LANGUAGE   '), chalk.gray('Language code'));
  console.log(chalk.cyan('    OPENAI_API_KEY           '), chalk.gray('OpenAI verification'));
  console.log(chalk.cyan('    ANTHROPIC_API_KEY        '), chalk.gray('Anthropic verification'));
  console.log(chalk.cyan('    GOOGLE_API_KEY           '), chalk.gray('Google AI verification\n'));
}

/**
 * Command suggestion for typos
 */
export function suggestCommand(input: string): string | null {
  const commands = [
    'analyze', 'verify', 'provider', 'config', 'history',
    'export', 'help', 'version'
  ];

  // Simple Levenshtein distance calculation
  const distance = (a: string, b: string): number => {
    const matrix: number[][] = [];

    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }

    return matrix[b.length][a.length];
  };

  // Find closest command
  let closestCommand = '';
  let minDistance = Infinity;

  for (const cmd of commands) {
    const dist = distance(input.toLowerCase(), cmd.toLowerCase());
    if (dist < minDistance && dist <= 3) {
      minDistance = dist;
      closestCommand = cmd;
    }
  }

  return closestCommand || null;
}

/**
 * Interactive tutorial mode
 */
export async function runTutorial(): Promise<void> {
  console.log(chalk.cyan(LOGO));
  console.log(chalk.bold.white('\n🎓 INTERACTIVE TUTORIAL\n'));
  console.log(chalk.gray('Welcome! This tutorial will guide you through using Nova Medicina.\n'));

  const steps = [
    {
      title: 'Understanding Nova Medicina',
      content:
        'Nova Medicina is an AI-powered medical analysis tool that helps you:\n' +
        '  • Analyze symptoms and assess urgency\n' +
        '  • Verify diagnoses with multiple AI models\n' +
        '  • Communicate with healthcare providers\n' +
        '  • Track your medical history\n\n' +
        'IMPORTANT: This tool provides information only. Always seek\n' +
        'professional medical care for health concerns.'
    },
    {
      title: 'Initial Setup',
      content:
        'First, let\'s configure your API key:\n\n' +
        chalk.green('  nova-medicina config --api-key YOUR_KEY_HERE\n\n') +
        'Get your free API key at: https://nova-medicina.io/api\n\n' +
        'Optional: Add AI verification keys for multi-model consensus:\n' +
        chalk.dim('  nova-medicina config --openai-key YOUR_OPENAI_KEY')
    },
    {
      title: 'Adding Healthcare Providers',
      content:
        'Add your healthcare provider for notifications:\n\n' +
        chalk.green('  nova-medicina provider add \\\n' +
        '    --name "Dr. Jane Smith" \\\n' +
        '    --email "dr.smith@clinic.com" \\\n' +
        '    --phone "555-0100" \\\n' +
        '    --primary\n\n') +
        'You can add multiple providers for different specialties.'
    },
    {
      title: 'Analyzing Symptoms',
      content:
        'Now let\'s analyze symptoms. Be specific and include:\n' +
        '  • What you\'re experiencing\n' +
        '  • When it started\n' +
        '  • Severity level\n' +
        '  • Any relevant medical history\n\n' +
        'Example:\n' +
        chalk.green('  nova-medicina analyze \\\n' +
        '    --age 45 \\\n' +
        '    --conditions "diabetes" \\\n' +
        '    "headache for 2 days, light sensitivity, nausea"')
    },
    {
      title: 'Verifying Analysis',
      content:
        'Get a second opinion from multiple AI models:\n\n' +
        chalk.green('  nova-medicina verify abc123def456 --explain\n\n') +
        'Verification provides:\n' +
        '  • Confidence score (0-100%)\n' +
        '  • Model consensus\n' +
        '  • Detailed reasoning\n' +
        '  • Discrepancy analysis'
    },
    {
      title: 'Notifying Providers',
      content:
        'Send your analysis to healthcare providers:\n\n' +
        chalk.green('  nova-medicina provider notify abc123def456 \\\n' +
        '    --urgent \\\n' +
        '    --request-callback\n\n') +
        'They\'ll receive:\n' +
        '  • Your symptom analysis\n' +
        '  • AI assessment and confidence\n' +
        '  • Your contact information\n' +
        '  • Any supporting photos/data'
    },
    {
      title: 'Emergency Situations',
      content:
        chalk.red.bold('⚠️  CRITICAL: For Medical Emergencies ⚠️\n\n') +
        'If you experience any of these, call 911 immediately:\n' +
        '  • Chest pain or pressure\n' +
        '  • Difficulty breathing\n' +
        '  • Severe bleeding\n' +
        '  • Loss of consciousness\n' +
        '  • Stroke symptoms\n' +
        '  • Severe allergic reaction\n\n' +
        chalk.yellow('Nova Medicina is NOT for emergencies. Always call 911 first.')
    },
    {
      title: 'Best Practices',
      content:
        'Tips for effective use:\n\n' +
        '  ✓ Be detailed and specific with symptoms\n' +
        '  ✓ Include relevant medical history\n' +
        '  ✓ Use verification for important decisions\n' +
        '  ✓ Keep provider information updated\n' +
        '  ✓ Save analysis history for trends\n' +
        '  ✓ Include photos when relevant\n\n' +
        '  ✗ Don\'t use for emergencies\n' +
        '  ✗ Don\'t replace doctor visits\n' +
        '  ✗ Don\'t share API keys\n' +
        '  ✗ Don\'t ignore severe symptoms'
    }
  ];

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    console.log(boxen(
      chalk.bold.cyan(`Step ${i + 1}/${steps.length}: ${step.title}\n\n`) +
      chalk.white(step.content),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'round',
        borderColor: 'cyan'
      }
    ));

    if (i < steps.length - 1) {
      console.log(chalk.dim('Press Enter to continue...'));
      // In real implementation, would await user input
      console.log('');
    }
  }

  console.log(chalk.bold.green('\n✓ Tutorial Complete!\n'));
  console.log(chalk.gray('You\'re ready to use Nova Medicina. For more help:\n'));
  console.log(chalk.cyan('  nova-medicina --help'));
  console.log(chalk.cyan('  nova-medicina <command> --help'));
  console.log(chalk.cyan('  https://nova-medicina.io/docs\n'));
}

/**
 * Show context-sensitive help based on partial command
 */
export function showContextHelp(context: string[]): void {
  if (context.length === 0) {
    showMainHelp();
    return;
  }

  const command = context[0].toLowerCase();

  switch (command) {
    case 'analyze':
      showAnalyzeHelp();
      break;
    case 'verify':
      showVerifyHelp();
      break;
    case 'provider':
      showProviderHelp();
      break;
    case 'config':
      showConfigHelp();
      break;
    default:
      const suggestion = suggestCommand(command);
      console.log(chalk.red(`\n❌ Unknown command: ${command}\n`));
      if (suggestion) {
        console.log(chalk.yellow(`Did you mean: ${chalk.green(suggestion)}?\n`));
        console.log(chalk.gray(`Try: nova-medicina help ${suggestion}\n`));
      } else {
        console.log(chalk.gray('Run'), chalk.cyan('nova-medicina --help'), chalk.gray('to see all commands.\n'));
      }
  }
}

/**
 * Display provider contact information
 */
export function showProviderContacts(providers: any[]): void {
  console.log(chalk.bold.white('\n👨‍⚕️  YOUR HEALTHCARE PROVIDERS\n'));

  if (providers.length === 0) {
    console.log(chalk.yellow('No providers configured.\n'));
    console.log(chalk.gray('Add a provider with:\n'));
    console.log(chalk.cyan('  nova-medicina provider add --name "Dr. Smith" --email "dr.smith@clinic.com"\n'));
    return;
  }

  providers.forEach((provider, index) => {
    const isPrimary = provider.primary ? chalk.green('★ PRIMARY') : '';
    const isEmergency = provider.emergency ? chalk.red('🚨 EMERGENCY') : '';

    console.log(boxen(
      chalk.bold.cyan(`${provider.name}`) + ' ' + isPrimary + ' ' + isEmergency + '\n\n' +
      (provider.specialty ? chalk.gray('Specialty: ') + chalk.white(provider.specialty) + '\n' : '') +
      (provider.clinic ? chalk.gray('Clinic: ') + chalk.white(provider.clinic) + '\n' : '') +
      (provider.email ? chalk.gray('Email: ') + chalk.blue(provider.email) + '\n' : '') +
      (provider.phone ? chalk.gray('Phone: ') + chalk.white(provider.phone) + '\n' : '') +
      (provider.fax ? chalk.gray('Fax: ') + chalk.white(provider.fax) : ''),
      {
        padding: 1,
        margin: { top: 0, bottom: 1, left: 2, right: 0 },
        borderStyle: 'round',
        borderColor: provider.primary ? 'green' : provider.emergency ? 'red' : 'white'
      }
    ));
  });
}

/**
 * Export help system functions
 */
export const helpSystem = {
  showMainHelp,
  showAnalyzeHelp,
  showVerifyHelp,
  showProviderHelp,
  showConfigHelp,
  suggestCommand,
  runTutorial,
  showContextHelp,
  showProviderContacts,
  LOGO,
  SAFETY_WARNING
};

export default helpSystem;
