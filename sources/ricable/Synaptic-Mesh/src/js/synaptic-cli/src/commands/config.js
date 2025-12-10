import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';
import { loadConfig, saveConfig } from '../config/loader.js';

export function configCommand() {
  const cmd = new Command('config');
  
  cmd
    .description('Manage Synaptic Neural Mesh configuration');
  
  // Get configuration value
  cmd
    .command('get [key]')
    .description('Get configuration value(s)')
    .option('-j, --json', 'Output as JSON')
    .action(async (key, options) => {
      try {
        const config = await loadConfig();
        if (!config) {
          console.log(chalk.red('No configuration found. Run "synaptic init" first.'));
          return;
        }
        
        if (key) {
          const value = getNestedValue(config, key);
          if (value === undefined) {
            console.log(chalk.red(`Configuration key '${key}' not found.`));
            return;
          }
          
          if (options.json) {
            console.log(JSON.stringify(value, null, 2));
          } else {
            console.log(value);
          }
        } else {
          if (options.json) {
            console.log(JSON.stringify(config, null, 2));
          } else {
            displayConfig(config);
          }
        }
      } catch (error) {
        console.error(chalk.red('Failed to get config: ' + error.message));
      }
    });
  
  // Set configuration value
  cmd
    .command('set <key> <value>')
    .description('Set configuration value')
    .action(async (key, value, options) => {
      const spinner = ora('Updating configuration...').start();
      
      try {
        const config = await loadConfig();
        if (!config) {
          spinner.fail('No configuration found. Run "synaptic init" first.');
          return;
        }
        
        // Parse value (try JSON first, then string)
        let parsedValue;
        try {
          parsedValue = JSON.parse(value);
        } catch {
          parsedValue = value;
        }
        
        setNestedValue(config, key, parsedValue);
        await saveConfig(config);
        
        spinner.succeed(chalk.green(`Configuration updated: ${key} = ${JSON.stringify(parsedValue)}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to set config: ' + error.message));
      }
    });
  
  // List all configuration keys
  cmd
    .command('list')
    .alias('ls')
    .description('List all configuration keys')
    .action(async () => {
      try {
        const config = await loadConfig();
        if (!config) {
          console.log(chalk.red('No configuration found. Run "synaptic init" first.'));
          return;
        }
        
        const keys = getAllKeys(config);
        console.log(chalk.cyan('Available configuration keys:\n'));
        
        keys.forEach(key => {
          const value = getNestedValue(config, key);
          const valueStr = typeof value === 'object' ? '[object]' : String(value);
          console.log(`  ${chalk.white(key)} = ${chalk.gray(valueStr)}`);
        });
        
      } catch (error) {
        console.error(chalk.red('Failed to list config: ' + error.message));
      }
    });
  
  // Reset configuration
  cmd
    .command('reset')
    .description('Reset configuration to defaults')
    .option('-f, --force', 'Force reset without confirmation')
    .action(async (options) => {
      try {
        if (!options.force) {
          const { confirm } = await inquirer.prompt([
            {
              type: 'confirm',
              name: 'confirm',
              message: 'Are you sure you want to reset configuration to defaults?',
              default: false
            }
          ]);
          
          if (!confirm) {
            console.log(chalk.yellow('Reset cancelled.'));
            return;
          }
        }
        
        const spinner = ora('Resetting configuration...').start();
        
        // Create default config
        const { createDefaultConfig } = await import('../config/default.js');
        const defaultConfig = createDefaultConfig();
        
        await saveConfig(defaultConfig);
        
        spinner.succeed(chalk.green('Configuration reset to defaults.'));
        
      } catch (error) {
        console.error(chalk.red('Failed to reset config: ' + error.message));
      }
    });
  
  // Edit configuration interactively
  cmd
    .command('edit')
    .description('Edit configuration interactively')
    .action(async () => {
      try {
        const config = await loadConfig();
        if (!config) {
          console.log(chalk.red('No configuration found. Run "synaptic init" first.'));
          return;
        }
        
        console.log(chalk.cyan('Interactive Configuration Editor\n'));
        
        const { section } = await inquirer.prompt([
          {
            type: 'list',
            name: 'section',
            message: 'Select configuration section to edit:',
            choices: [
              { name: 'Project Settings', value: 'project' },
              { name: 'Mesh Configuration', value: 'mesh' },
              { name: 'Neural Network', value: 'neural' },
              { name: 'P2P Network', value: 'peer' },
              { name: 'Features', value: 'features' },
              { name: 'Advanced', value: 'advanced' }
            ]
          }
        ]);
        
        await editSection(config, section);
        
        const spinner = ora('Saving configuration...').start();
        await saveConfig(config);
        spinner.succeed(chalk.green('Configuration saved successfully.'));
        
      } catch (error) {
        console.error(chalk.red('Failed to edit config: ' + error.message));
      }
    });
    
  return cmd;
}

function displayConfig(config) {
  console.log(chalk.cyan('Synaptic Neural Mesh Configuration:\n'));
  
  console.log(chalk.white('📁 Project:'));
  console.log(`  Name: ${config.project.name}`);
  console.log(`  Template: ${config.project.template}`);
  
  console.log(chalk.white('\n🕸️  Mesh:'));
  console.log(`  Topology: ${config.mesh.topology}`);
  console.log(`  Default Agents: ${config.mesh.defaultAgents}`);
  
  console.log(chalk.white('\n🧠 Neural:'));
  console.log(`  Enabled: ${config.neural?.enabled || false}`);
  if (config.neural?.defaultModel) {
    console.log(`  Default Model: ${config.neural.defaultModel}`);
  }
  
  console.log(chalk.white('\n🌐 Peer Network:'));
  console.log(`  Auto Discovery: ${config.peer.autoDiscovery}`);
  console.log(`  Max Peers: ${config.peer.maxPeers || 50}`);
  
  console.log(chalk.white('\n⚡ Features:'));
  console.log(`  MCP: ${config.features.mcp}`);
  console.log(`  WebUI: ${config.features.webui || false}`);
  console.log(`  Monitoring: ${config.features.monitoring || false}`);
}

async function editSection(config, section) {
  switch (section) {
    case 'project':
      await editProjectSection(config);
      break;
    case 'mesh':
      await editMeshSection(config);
      break;
    case 'neural':
      await editNeuralSection(config);
      break;
    case 'peer':
      await editPeerSection(config);
      break;
    case 'features':
      await editFeaturesSection(config);
      break;
    case 'advanced':
      await editAdvancedSection(config);
      break;
  }
}

async function editProjectSection(config) {
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'name',
      message: 'Project name:',
      default: config.project.name
    },
    {
      type: 'list',
      name: 'template',
      message: 'Project template:',
      choices: ['basic', 'advanced', 'enterprise'],
      default: config.project.template
    }
  ]);
  
  config.project.name = answers.name;
  config.project.template = answers.template;
}

async function editMeshSection(config) {
  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'topology',
      message: 'Mesh topology:',
      choices: ['mesh', 'hierarchical', 'ring', 'star'],
      default: config.mesh.topology
    },
    {
      type: 'number',
      name: 'defaultAgents',
      message: 'Default number of agents:',
      default: config.mesh.defaultAgents,
      validate: (input) => input > 0 && input <= 100
    }
  ]);
  
  config.mesh.topology = answers.topology;
  config.mesh.defaultAgents = answers.defaultAgents;
}

async function editNeuralSection(config) {
  if (!config.neural) config.neural = {};
  
  const answers = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'enabled',
      message: 'Enable neural network features:',
      default: config.neural.enabled || false
    }
  ]);
  
  config.neural.enabled = answers.enabled;
  
  if (answers.enabled) {
    const neuralAnswers = await inquirer.prompt([
      {
        type: 'input',
        name: 'defaultModel',
        message: 'Default model architecture:',
        default: config.neural.defaultModel || 'mlp'
      }
    ]);
    
    config.neural.defaultModel = neuralAnswers.defaultModel;
  }
}

async function editPeerSection(config) {
  const answers = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'autoDiscovery',
      message: 'Enable automatic peer discovery:',
      default: config.peer.autoDiscovery
    },
    {
      type: 'number',
      name: 'maxPeers',
      message: 'Maximum number of peers:',
      default: config.peer.maxPeers || 50,
      validate: (input) => input > 0 && input <= 1000
    }
  ]);
  
  config.peer.autoDiscovery = answers.autoDiscovery;
  config.peer.maxPeers = answers.maxPeers;
}

async function editFeaturesSection(config) {
  const answers = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'mcp',
      message: 'Enable Model Context Protocol (MCP):',
      default: config.features.mcp
    },
    {
      type: 'confirm',
      name: 'webui',
      message: 'Enable Web UI:',
      default: config.features.webui || false
    },
    {
      type: 'confirm',
      name: 'monitoring',
      message: 'Enable monitoring:',
      default: config.features.monitoring || false
    }
  ]);
  
  Object.assign(config.features, answers);
}

async function editAdvancedSection(config) {
  if (!config.advanced) config.advanced = {};
  
  const answers = await inquirer.prompt([
    {
      type: 'number',
      name: 'workerThreads',
      message: 'Number of worker threads:',
      default: config.advanced.workerThreads || 4
    },
    {
      type: 'number',
      name: 'maxMemory',
      message: 'Maximum memory usage (MB):',
      default: config.advanced.maxMemory || 1024
    }
  ]);
  
  Object.assign(config.advanced, answers);
}

// Utility functions
function getNestedValue(obj, path) {
  return path.split('.').reduce((current, key) => current?.[key], obj);
}

function setNestedValue(obj, path, value) {
  const keys = path.split('.');
  const lastKey = keys.pop();
  const target = keys.reduce((current, key) => {
    if (!current[key] || typeof current[key] !== 'object') {
      current[key] = {};
    }
    return current[key];
  }, obj);
  target[lastKey] = value;
}

function getAllKeys(obj, prefix = '') {
  const keys = [];
  
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      keys.push(...getAllKeys(value, fullKey));
    } else {
      keys.push(fullKey);
    }
  }
  
  return keys.sort();
}