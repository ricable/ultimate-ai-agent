/**
 * Initialize command for Synaptic Neural Mesh
 * Creates a new project with all necessary configuration
 */

const chalk = require('chalk');
const ora = require('ora');
const fs = require('fs-extra');
const path = require('path');
const inquirer = require('inquirer');
const { spawn } = require('child_process');

const templates = {
  default: {
    name: 'Default',
    description: 'Basic Synaptic Neural Mesh setup'
  },
  enterprise: {
    name: 'Enterprise',
    description: 'Full-featured setup with Docker, K8s, and monitoring'
  },
  research: {
    name: 'Research',
    description: 'Optimized for neural network research and experimentation'
  },
  edge: {
    name: 'Edge Computing',
    description: 'Lightweight setup for edge devices and IoT'
  }
};

async function execute(projectName, options) {
  const spinner = ora();
  
  try {
    // Determine project directory
    const targetDir = projectName ? path.resolve(projectName) : process.cwd();
    const projectBaseName = path.basename(targetDir);
    
    // Check if directory exists and is not empty
    if (fs.existsSync(targetDir) && fs.readdirSync(targetDir).length > 0) {
      const { proceed } = await inquirer.prompt([{
        type: 'confirm',
        name: 'proceed',
        message: `Directory ${chalk.yellow(targetDir)} is not empty. Continue?`,
        default: false
      }]);
      
      if (!proceed) {
        console.log(chalk.yellow('Initialization cancelled'));
        return;
      }
    }
    
    // Create directory if it doesn't exist
    await fs.ensureDir(targetDir);
    
    // Get project configuration
    const config = await getProjectConfig(projectBaseName, options);
    
    spinner.start('Creating project structure...');
    
    // Create base directories
    const dirs = [
      'src',
      'src/nodes',
      'src/neural',
      'src/dag',
      'src/p2p',
      'config',
      'data',
      'scripts',
      'tests',
      'docs'
    ];
    
    if (options.docker) {
      dirs.push('docker');
    }
    
    if (options.k8s) {
      dirs.push('k8s', 'k8s/base', 'k8s/overlays');
    }
    
    for (const dir of dirs) {
      await fs.ensureDir(path.join(targetDir, dir));
    }
    
    spinner.succeed('Project structure created');
    
    // Create configuration files
    spinner.start('Creating configuration files...');
    
    // Package.json
    await createPackageJson(targetDir, config);
    
    // Synaptic config
    await createSynapticConfig(targetDir, config);
    
    // Docker files if requested
    if (options.docker) {
      await createDockerFiles(targetDir, config);
    }
    
    // Kubernetes files if requested
    if (options.k8s) {
      await createK8sFiles(targetDir, config);
    }
    
    // Create example files
    await createExampleFiles(targetDir, config);
    
    spinner.succeed('Configuration files created');
    
    // Install dependencies
    if (!options.noInstall) {
      spinner.start('Installing dependencies...');
      await installDependencies(targetDir);
      spinner.succeed('Dependencies installed');
    }
    
    // Display success message
    console.log('\n' + chalk.green('✨ Project initialized successfully!'));
    console.log('\nNext steps:');
    
    if (projectName) {
      console.log(chalk.cyan(`  cd ${projectName}`));
    }
    
    console.log(chalk.cyan('  synaptic start          # Start the node'));
    console.log(chalk.cyan('  synaptic status         # Check node status'));
    console.log(chalk.cyan('  synaptic peer list      # View connected peers'));
    
    console.log('\n' + chalk.gray('Documentation: https://github.com/synaptic-neural-mesh/docs'));
    
  } catch (error) {
    spinner.fail('Initialization failed');
    console.error(chalk.red('Error:'), error.message);
    process.exit(1);
  }
}

async function getProjectConfig(projectName, options) {
  if (options.template && options.template !== 'default') {
    return {
      name: projectName,
      template: options.template,
      ...templates[options.template]
    };
  }
  
  // Interactive configuration
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'name',
      message: 'Project name:',
      default: projectName
    },
    {
      type: 'input',
      name: 'description',
      message: 'Project description:',
      default: 'A Synaptic Neural Mesh node'
    },
    {
      type: 'list',
      name: 'template',
      message: 'Select a template:',
      choices: Object.entries(templates).map(([key, value]) => ({
        name: `${value.name} - ${value.description}`,
        value: key
      }))
    },
    {
      type: 'checkbox',
      name: 'features',
      message: 'Select additional features:',
      choices: [
        { name: 'Web UI Dashboard', value: 'webui', checked: true },
        { name: 'Prometheus Metrics', value: 'metrics', checked: true },
        { name: 'Advanced Neural Models', value: 'neural', checked: false },
        { name: 'GPU Support', value: 'gpu', checked: false },
        { name: 'Auto-scaling', value: 'autoscale', checked: false }
      ]
    }
  ]);
  
  return answers;
}

async function createPackageJson(targetDir, config) {
  const packageJson = {
    name: config.name,
    version: '0.1.0',
    description: config.description || 'A Synaptic Neural Mesh node',
    main: 'src/index.js',
    scripts: {
      start: 'synaptic start',
      'start:dev': 'synaptic start --dev',
      test: 'jest',
      'test:watch': 'jest --watch',
      lint: 'eslint src',
      format: 'prettier --write src'
    },
    keywords: ['synaptic', 'neural-mesh', 'distributed-ai'],
    author: '',
    license: 'MIT',
    dependencies: {
      'synaptic-mesh': '^1.0.0-alpha.1'
    },
    devDependencies: {
      'eslint': '^8.56.0',
      'jest': '^29.7.0',
      'prettier': '^3.1.1'
    }
  };
  
  await fs.writeJson(path.join(targetDir, 'package.json'), packageJson, { spaces: 2 });
}

async function createSynapticConfig(targetDir, config) {
  const synapticConfig = {
    version: '1.0',
    node: {
      id: null, // Will be generated on first run
      name: config.name,
      port: 7890,
      host: '0.0.0.0'
    },
    mesh: {
      topology: 'mesh',
      maxPeers: 50,
      discoveryInterval: 30000
    },
    neural: {
      models: ['transformer', 'lstm', 'gnn'],
      defaultModel: 'transformer',
      maxConcurrentInference: 10
    },
    dag: {
      consensusAlgorithm: 'proof-of-stake',
      blockTime: 5000,
      maxTransactionsPerBlock: 1000
    },
    storage: {
      dataDir: './data',
      maxSize: '10GB',
      cacheSize: '1GB'
    },
    features: config.features || []
  };
  
  await fs.writeJson(
    path.join(targetDir, 'config', 'synaptic.config.json'),
    synapticConfig,
    { spaces: 2 }
  );
}

async function createDockerFiles(targetDir, config) {
  const dockerfile = `FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 7890 8080

CMD ["npm", "start"]
`;
  
  const dockerCompose = `version: '3.8'

services:
  synaptic-node:
    build: .
    ports:
      - "7890:7890"
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - NODE_ENV=production
      - SYNAPTIC_CONFIG=/app/config/synaptic.config.json
    restart: unless-stopped
`;
  
  await fs.writeFile(path.join(targetDir, 'Dockerfile'), dockerfile);
  await fs.writeFile(path.join(targetDir, 'docker-compose.yml'), dockerCompose);
}

async function createK8sFiles(targetDir, config) {
  // Add Kubernetes manifests
  const deployment = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: synaptic-node
spec:
  replicas: 3
  selector:
    matchLabels:
      app: synaptic
  template:
    metadata:
      labels:
        app: synaptic
    spec:
      containers:
      - name: synaptic
        image: synaptic-mesh:latest
        ports:
        - containerPort: 7890
        - containerPort: 8080
`;
  
  await fs.writeFile(
    path.join(targetDir, 'k8s', 'base', 'deployment.yaml'),
    deployment
  );
}

async function createExampleFiles(targetDir, config) {
  // Create a simple example node
  const exampleNode = `const { SynapticNode } = require('synaptic-mesh');

async function main() {
  const node = new SynapticNode({
    configPath: './config/synaptic.config.json'
  });
  
  await node.start();
  
  console.log('Synaptic node started!');
  console.log('Node ID:', node.id);
  console.log('Listening on:', node.address);
}

main().catch(console.error);
`;
  
  await fs.writeFile(path.join(targetDir, 'src', 'index.js'), exampleNode);
}

async function installDependencies(targetDir) {
  return new Promise((resolve, reject) => {
    const npm = spawn('npm', ['install'], {
      cwd: targetDir,
      stdio: 'ignore'
    });
    
    npm.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`npm install exited with code ${code}`));
      } else {
        resolve();
      }
    });
    
    npm.on('error', reject);
  });
}

module.exports = { execute };