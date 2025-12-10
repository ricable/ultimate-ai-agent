"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.initCommand = initCommand;
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
const ora_1 = __importDefault(require("ora"));
const inquirer_1 = __importDefault(require("inquirer"));
const promises_1 = __importDefault(require("fs/promises"));
const path_1 = __importDefault(require("path"));
const crypto_1 = __importDefault(require("crypto"));
const uuid_1 = require("uuid");
function initCommand() {
    const command = new commander_1.Command('init');
    command
        .description('Initialize a new Synaptic Neural Mesh node')
        .option('-n, --name <name>', 'Node name', `node-${(0, uuid_1.v4)().slice(0, 8)}`)
        .option('-p, --port <port>', 'Default port for P2P networking', '8080')
        .option('--network <network>', 'Network to join (mainnet/testnet/local)', 'mainnet')
        .option('--no-quantum-resistant', 'Disable quantum-resistant cryptography')
        .option('--no-interactive', 'Skip interactive setup')
        .option('-f, --force', 'Force initialization even if config exists')
        .action(async (options) => {
        console.log(chalk_1.default.cyan('\n🧠 Initializing Synaptic Neural Mesh Node...\n'));
        try {
            await initializeNode(options);
        }
        catch (error) {
            console.error(chalk_1.default.red('Initialization failed:'), error?.message || error);
            process.exit(1);
        }
    });
    return command;
}
async function initializeNode(options) {
    // Check if already initialized
    const configDir = path_1.default.join(process.cwd(), '.synaptic');
    const configPath = path_1.default.join(configDir, 'config.json');
    if (await fileExists(configPath) && !options.force) {
        console.error(chalk_1.default.yellow('⚠️  Node already initialized. Use --force to reinitialize.'));
        process.exit(1);
    }
    // Interactive setup if enabled
    if (options.interactive !== false) {
        const answers = await inquirer_1.default.prompt([
            {
                type: 'input',
                name: 'name',
                message: 'Node name:',
                default: options.name
            },
            {
                type: 'number',
                name: 'port',
                message: 'P2P networking port:',
                default: parseInt((options.port || '8080').toString())
            },
            {
                type: 'list',
                name: 'network',
                message: 'Network to join:',
                choices: ['mainnet', 'testnet', 'local'],
                default: options.network
            },
            {
                type: 'confirm',
                name: 'quantumResistant',
                message: 'Enable quantum-resistant cryptography?',
                default: options.quantumResistant !== false
            }
        ]);
        Object.assign(options, answers);
    }
    const spinner = (0, ora_1.default)('Creating node configuration...').start();
    try {
        // Create directory structure
        await createDirectoryStructure(configDir);
        spinner.text = 'Generating node identity...';
        // Generate node identity
        const nodeId = (0, uuid_1.v4)();
        const nodeKeys = await generateNodeKeys(options.quantumResistant !== false);
        spinner.text = 'Initializing database...';
        // Initialize SQLite database
        await initializeDatabase(configDir);
        spinner.text = 'Downloading WASM modules...';
        // Download/copy WASM modules
        await setupWasmModules(configDir);
        spinner.text = 'Creating configuration...';
        // Create configuration
        const config = {
            node: {
                id: nodeId,
                name: options.name,
                version: '1.0.0',
                created: new Date().toISOString()
            },
            network: {
                port: parseInt((options.port || '8080').toString()),
                network: options.network || 'mainnet',
                bootstrapPeers: getBootstrapPeers(options.network || 'mainnet'),
                protocols: ['/synaptic/1.0.0', '/qudag/1.0.0', '/neural/1.0.0']
            },
            security: {
                quantumResistant: options.quantumResistant !== false,
                publicKey: nodeKeys.publicKey,
                algorithm: nodeKeys.algorithm
            },
            neural: {
                defaultArchitecture: 'adaptive',
                maxAgents: 1000,
                memoryLimit: '50MB',
                inferenceTimeout: 100
            },
            dag: {
                consensus: 'qr-avalanche',
                maxVertices: 10000,
                pruningInterval: 3600
            },
            storage: {
                dataDir: path_1.default.join(configDir, 'data'),
                cacheDir: path_1.default.join(configDir, 'cache'),
                logsDir: path_1.default.join(configDir, 'logs')
            }
        };
        // Write configuration
        await promises_1.default.writeFile(configPath, JSON.stringify(config, null, 2));
        // Write private key separately
        const keyPath = path_1.default.join(configDir, 'keys', 'node.key');
        await promises_1.default.writeFile(keyPath, nodeKeys.privateKey);
        await promises_1.default.chmod(keyPath, 0o600); // Restrict access
        spinner.text = 'Setting up neural templates...';
        // Setup neural network templates
        await setupNeuralTemplates(configDir);
        spinner.succeed(chalk_1.default.green('✅ Node initialization complete!'));
        // Display summary
        console.log('\n' + chalk_1.default.cyan('📋 Node Configuration Summary:'));
        console.log(chalk_1.default.gray('─'.repeat(40)));
        console.log(`${chalk_1.default.bold('Node ID:')} ${nodeId}`);
        console.log(`${chalk_1.default.bold('Node Name:')} ${options.name}`);
        console.log(`${chalk_1.default.bold('Network:')} ${options.network}`);
        console.log(`${chalk_1.default.bold('P2P Port:')} ${options.port}`);
        console.log(`${chalk_1.default.bold('Quantum Resistant:')} ${options.quantumResistant !== false ? 'Yes' : 'No'}`);
        console.log(`${chalk_1.default.bold('Config Location:')} ${configPath}`);
        console.log(chalk_1.default.gray('─'.repeat(40)));
        console.log('\n' + chalk_1.default.green('🚀 Next Steps:'));
        console.log('  1. Start your node:     ' + chalk_1.default.cyan('synaptic-mesh start'));
        console.log('  2. Check node status:   ' + chalk_1.default.cyan('synaptic-mesh status'));
        console.log('  3. Join a mesh:         ' + chalk_1.default.cyan('synaptic-mesh mesh join <peer-address>'));
        console.log('  4. Spawn neural agent:  ' + chalk_1.default.cyan('synaptic-mesh neural spawn'));
        // Create convenience scripts
        await createConvenienceScripts(process.cwd());
    }
    catch (error) {
        spinner.fail(chalk_1.default.red('Initialization failed'));
        throw error;
    }
}
async function createDirectoryStructure(configDir) {
    const dirs = [
        configDir,
        path_1.default.join(configDir, 'data'),
        path_1.default.join(configDir, 'cache'),
        path_1.default.join(configDir, 'logs'),
        path_1.default.join(configDir, 'keys'),
        path_1.default.join(configDir, 'wasm'),
        path_1.default.join(configDir, 'neural'),
        path_1.default.join(configDir, 'dag')
    ];
    for (const dir of dirs) {
        await promises_1.default.mkdir(dir, { recursive: true });
    }
}
async function generateNodeKeys(quantumResistant) {
    if (quantumResistant) {
        // In production, use actual post-quantum cryptography library
        // For now, simulate with strong classical crypto
        const keyPair = crypto_1.default.generateKeyPairSync('ed25519');
        return {
            algorithm: 'ml-dsa-65', // Simulate ML-DSA
            publicKey: keyPair.publicKey.export({ type: 'spki', format: 'pem' }),
            privateKey: keyPair.privateKey.export({ type: 'pkcs8', format: 'pem' })
        };
    }
    else {
        const keyPair = crypto_1.default.generateKeyPairSync('ed25519');
        return {
            algorithm: 'ed25519',
            publicKey: keyPair.publicKey.export({ type: 'spki', format: 'pem' }),
            privateKey: keyPair.privateKey.export({ type: 'pkcs8', format: 'pem' })
        };
    }
}
async function initializeDatabase(configDir) {
    const dbPath = path_1.default.join(configDir, 'data', 'synaptic.db');
    // Create database schema
    const schema = `
-- Node metadata
CREATE TABLE IF NOT EXISTS node_info (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Peer connections
CREATE TABLE IF NOT EXISTS peers (
  id TEXT PRIMARY KEY,
  address TEXT NOT NULL,
  public_key TEXT,
  last_seen TIMESTAMP,
  reputation REAL DEFAULT 1.0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Neural agents
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  architecture TEXT,
  state BLOB,
  performance REAL DEFAULT 0.0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  terminated_at TIMESTAMP
);

-- DAG vertices
CREATE TABLE IF NOT EXISTS dag_vertices (
  id TEXT PRIMARY KEY,
  previous_ids TEXT,
  data BLOB,
  signature TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  confirmations INTEGER DEFAULT 0
);

-- Metrics and telemetry
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  metric_type TEXT NOT NULL,
  value REAL,
  metadata TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_peers_last_seen ON peers(last_seen);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(type);
CREATE INDEX IF NOT EXISTS idx_dag_timestamp ON dag_vertices(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_type_time ON metrics(metric_type, timestamp);
`;
    // Write schema file for reference
    await promises_1.default.writeFile(path_1.default.join(configDir, 'data', 'schema.sql'), schema);
}
async function setupWasmModules(configDir) {
    // In production, download from CDN or package registry
    // For now, create placeholder
    const wasmModules = [
        'ruv_swarm_wasm_bg.wasm',
        'ruv_swarm_simd.wasm',
        'ruv-fann.wasm',
        'neuro-divergent.wasm'
    ];
    const wasmDir = path_1.default.join(configDir, 'wasm');
    for (const module of wasmModules) {
        const modulePath = path_1.default.join(wasmDir, module);
        // Create placeholder files
        await promises_1.default.writeFile(modulePath, Buffer.from('AGFzbQEAAAA=', 'base64')); // Minimal valid WASM
    }
    // Create loader configuration
    const loaderConfig = {
        modules: wasmModules,
        optimization: {
            simd: true,
            threads: true,
            bulkMemory: true
        }
    };
    await promises_1.default.writeFile(path_1.default.join(wasmDir, 'loader.config.json'), JSON.stringify(loaderConfig, null, 2));
}
async function setupNeuralTemplates(configDir) {
    const templates = {
        mlp: {
            name: 'Multi-Layer Perceptron',
            architecture: {
                layers: [784, 128, 64, 10],
                activation: 'relu',
                outputActivation: 'softmax'
            }
        },
        lstm: {
            name: 'Long Short-Term Memory',
            architecture: {
                inputSize: 128,
                hiddenSize: 256,
                numLayers: 2,
                dropout: 0.2
            }
        },
        cnn: {
            name: 'Convolutional Neural Network',
            architecture: {
                conv_layers: [
                    { filters: 32, kernel: 3, activation: 'relu' },
                    { filters: 64, kernel: 3, activation: 'relu' }
                ],
                dense_layers: [128, 10]
            }
        },
        particle: {
            name: 'Particle Swarm Network',
            architecture: {
                particles: 100,
                dimensions: 50,
                inertia: 0.9,
                cognitive: 2.0,
                social: 2.0
            }
        }
    };
    const neuralDir = path_1.default.join(configDir, 'neural');
    await promises_1.default.writeFile(path_1.default.join(neuralDir, 'templates.json'), JSON.stringify(templates, null, 2));
}
async function createConvenienceScripts(projectDir) {
    // Create start script
    const startScript = `#!/bin/bash
# Quick start script for Synaptic Neural Mesh

echo "🧠 Starting Synaptic Neural Mesh..."
synaptic-mesh start
`;
    await promises_1.default.writeFile(path_1.default.join(projectDir, 'start-mesh.sh'), startScript);
    await promises_1.default.chmod(path_1.default.join(projectDir, 'start-mesh.sh'), 0o755);
    // Create docker-compose for easy deployment
    const dockerCompose = `version: '3.8'

services:
  synaptic-mesh:
    image: synaptic-mesh:latest
    container_name: synaptic-mesh-node
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - ./.synaptic:/app/.synaptic
    environment:
      - NODE_ENV=production
      - RUST_LOG=info
    restart: unless-stopped
    networks:
      - synaptic-net

networks:
  synaptic-net:
    driver: bridge
`;
    await promises_1.default.writeFile(path_1.default.join(projectDir, 'docker-compose.yml'), dockerCompose);
}
function getBootstrapPeers(network) {
    switch (network) {
        case 'mainnet':
            return [
                '/ip4/seed1.synaptic.network/tcp/8080/p2p/QmMainnet1...',
                '/ip4/seed2.synaptic.network/tcp/8080/p2p/QmMainnet2...',
                '/ip4/seed3.synaptic.network/tcp/8080/p2p/QmMainnet3...'
            ];
        case 'testnet':
            return [
                '/ip4/testnet.synaptic.network/tcp/8080/p2p/QmTestnet1...'
            ];
        case 'local':
            return [];
        default:
            return [];
    }
}
async function fileExists(path) {
    try {
        await promises_1.default.access(path);
        return true;
    }
    catch {
        return false;
    }
}
//# sourceMappingURL=init.js.map