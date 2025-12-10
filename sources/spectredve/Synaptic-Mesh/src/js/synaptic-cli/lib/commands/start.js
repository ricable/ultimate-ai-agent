"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.startCommand = startCommand;
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
const ora_1 = __importDefault(require("ora"));
const path_1 = __importDefault(require("path"));
const promises_1 = __importDefault(require("fs/promises"));
const http_1 = require("http");
const blessed_1 = __importDefault(require("blessed"));
const blessed_contrib_1 = __importDefault(require("blessed-contrib"));
function startCommand() {
    const command = new commander_1.Command('start');
    command
        .description('Start the Synaptic Neural Mesh node')
        .option('-p, --port <port>', 'Override configured P2P port')
        .option('-d, --daemon', 'Run as background daemon')
        .option('--no-ui', 'Disable terminal UI')
        .option('--metrics <port>', 'Enable metrics server on specified port')
        .action(async (options) => {
        try {
            await startNode(options);
        }
        catch (error) {
            console.error(chalk_1.default.red('Failed to start node:'), error?.message || error);
            process.exit(1);
        }
    });
    return command;
}
async function startNode(options) {
    const spinner = (0, ora_1.default)('Starting Synaptic Neural Mesh node...').start();
    try {
        // Load configuration
        const configPath = path_1.default.join(process.cwd(), '.synaptic', 'config.json');
        const config = JSON.parse(await promises_1.default.readFile(configPath, 'utf-8'));
        // Override port if specified
        if (options.port) {
            config.network.port = parseInt(options.port.toString());
        }
        spinner.text = 'Initializing P2P network...';
        // Initialize networking
        const p2pNode = await initializeP2PNode(config);
        spinner.text = 'Loading WASM modules...';
        // Load WASM modules
        await loadWasmModules(config);
        spinner.text = 'Starting neural engine...';
        // Initialize neural engine
        const neuralEngine = await initializeNeuralEngine(config);
        spinner.text = 'Connecting to DAG network...';
        // Initialize DAG consensus
        const dagNode = await initializeDAGNode(config);
        spinner.succeed(chalk_1.default.green('✅ Node started successfully!'));
        // Display connection info
        console.log('\n' + chalk_1.default.cyan('📡 Node Information:'));
        console.log(chalk_1.default.gray('─'.repeat(40)));
        console.log(`${chalk_1.default.bold('Node ID:')} ${config.node.id}`);
        console.log(`${chalk_1.default.bold('P2P Address:')} /ip4/0.0.0.0/tcp/${config.network.port}`);
        console.log(`${chalk_1.default.bold('Network:')} ${config.network.network}`);
        console.log(`${chalk_1.default.bold('Status:')} ${chalk_1.default.green('Online')}`);
        console.log(chalk_1.default.gray('─'.repeat(40)));
        if (options.daemon) {
            console.log(chalk_1.default.yellow('\n⚡ Running in daemon mode...'));
            console.log('Use ' + chalk_1.default.cyan('synaptic-mesh stop') + ' to stop the node');
            // Detach from terminal
            if (process.send) {
                process.send('ready');
            }
        }
        else if (options.ui !== false) {
            // Start terminal UI
            await startTerminalUI(config, p2pNode, neuralEngine, dagNode);
        }
        else {
            console.log(chalk_1.default.yellow('\n⚡ Node is running. Press Ctrl+C to stop.'));
            // Handle graceful shutdown
            process.on('SIGINT', async () => {
                console.log(chalk_1.default.yellow('\n\n🛑 Shutting down node...'));
                await shutdownNode(p2pNode, neuralEngine, dagNode);
                process.exit(0);
            });
        }
        // Start metrics server if requested
        if (options.metrics) {
            await startMetricsServer(parseInt(options.metrics.toString()));
        }
    }
    catch (error) {
        spinner.fail(chalk_1.default.red('Failed to start node'));
        throw error;
    }
}
async function initializeP2PNode(config) {
    // Simulated P2P initialization
    // In production, integrate with libp2p and QuDAG
    return {
        id: config.node.id,
        multiaddrs: [`/ip4/0.0.0.0/tcp/${config.network.port}`],
        peers: new Map(),
        protocols: config.network.protocols,
        async start() {
            // Start listening
        },
        async connect(peerAddr) {
            // Connect to peer
        },
        async broadcast(message) {
            // Broadcast to all peers
        },
        async stop() {
            // Stop P2P node
        }
    };
}
async function loadWasmModules(config) {
    const wasmDir = path_1.default.join(process.cwd(), '.synaptic', 'wasm');
    const modules = {};
    for (const moduleName of config.synaptic?.wasmModules || []) {
        try {
            // In production, actually load and instantiate WASM
            modules[moduleName] = {
                name: moduleName,
                loaded: true,
                exports: {}
            };
        }
        catch (error) {
            console.warn(chalk_1.default.yellow(`⚠️  Failed to load WASM module: ${moduleName}`));
        }
    }
    return modules;
}
async function initializeNeuralEngine(config) {
    return {
        agents: new Map(),
        templates: await loadNeuralTemplates(config),
        async spawnAgent(type, options) {
            const agentId = `agent-${Date.now()}`;
            const agent = {
                id: agentId,
                type,
                network: null, // Neural network instance
                performance: 0,
                tasks: []
            };
            this.agents.set(agentId, agent);
            return agent;
        },
        async terminateAgent(agentId) {
            this.agents.delete(agentId);
        },
        async processInput(agentId, input) {
            // Process input through neural network
            return { output: [], confidence: 0.95 };
        }
    };
}
async function initializeDAGNode(config) {
    return {
        vertices: new Map(),
        consensus: config.dag.consensus,
        async addVertex(data) {
            const vertex = {
                id: `vertex-${Date.now()}`,
                data,
                timestamp: new Date(),
                confirmations: 0
            };
            this.vertices.set(vertex.id, vertex);
            return vertex;
        },
        async queryVertex(id) {
            return this.vertices.get(id);
        },
        async getConsensusState() {
            return {
                vertices: this.vertices.size,
                confirmed: Array.from(this.vertices.values()).filter(v => v.confirmations > 6).length
            };
        }
    };
}
async function loadNeuralTemplates(config) {
    const templatesPath = path_1.default.join(process.cwd(), '.synaptic', 'neural', 'templates.json');
    try {
        return JSON.parse(await promises_1.default.readFile(templatesPath, 'utf-8'));
    }
    catch {
        return {};
    }
}
async function startTerminalUI(config, p2pNode, neuralEngine, dagNode) {
    // Create blessed screen
    const screen = blessed_1.default.screen({
        smartCSR: true,
        title: 'Synaptic Neural Mesh'
    });
    // Create grid layout
    const grid = new blessed_contrib_1.default.grid({ rows: 12, cols: 12, screen });
    // Network status widget
    const networkBox = grid.set(0, 0, 3, 6, blessed_1.default.box, {
        label: ' Network Status ',
        border: { type: 'line', fg: 'cyan' },
        style: { fg: 'white', border: { fg: 'cyan' } }
    });
    // Neural agents widget
    const agentsTable = grid.set(0, 6, 3, 6, blessed_contrib_1.default.table, {
        keys: true,
        fg: 'white',
        selectedFg: 'white',
        selectedBg: 'blue',
        interactive: true,
        label: ' Neural Agents ',
        border: { type: 'line', fg: 'cyan' },
        columnSpacing: 3,
        columnWidth: [10, 15, 10, 10]
    });
    // DAG visualization
    const dagChart = grid.set(3, 0, 5, 8, blessed_contrib_1.default.line, {
        style: { line: 'yellow', text: 'green', baseline: 'black' },
        xLabelPadding: 3,
        xPadding: 5,
        showLegend: true,
        legend: { width: 12 },
        label: ' DAG Activity '
    });
    // Metrics gauges
    const cpuGauge = grid.set(3, 8, 2, 2, blessed_contrib_1.default.gauge, {
        label: ' CPU ',
        stroke: 'green',
        fill: 'white'
    });
    const memGauge = grid.set(5, 8, 2, 2, blessed_contrib_1.default.gauge, {
        label: ' Memory ',
        stroke: 'yellow',
        fill: 'white'
    });
    const peersGauge = grid.set(3, 10, 2, 2, blessed_contrib_1.default.gauge, {
        label: ' Peers ',
        stroke: 'cyan',
        fill: 'white'
    });
    const agentsGauge = grid.set(5, 10, 2, 2, blessed_contrib_1.default.gauge, {
        label: ' Agents ',
        stroke: 'magenta',
        fill: 'white'
    });
    // Logs widget
    const logBox = grid.set(8, 0, 4, 12, blessed_contrib_1.default.log, {
        fg: 'green',
        selectedFg: 'green',
        label: ' Logs ',
        border: { type: 'line', fg: 'cyan' }
    });
    // Update functions
    const updateUI = () => {
        // Update network status
        networkBox.setContent(`Node ID: ${config.node.id.slice(0, 16)}...\n` +
            `Network: ${config.network.network}\n` +
            `Port: ${config.network.port}\n` +
            `Peers: ${p2pNode.peers.size}\n` +
            `Status: ${chalk_1.default.green('Online')}`);
        // Update agents table
        const agentData = Array.from(neuralEngine.agents.values()).map((agent) => [
            agent.id.slice(0, 10),
            agent.type,
            agent.tasks.length.toString(),
            `${(agent.performance * 100).toFixed(1)}%`
        ]);
        agentsTable.setData({
            headers: ['ID', 'Type', 'Tasks', 'Performance'],
            data: agentData
        });
        // Update gauges
        cpuGauge.setPercent(Math.random() * 100);
        memGauge.setPercent(Math.random() * 100);
        peersGauge.setPercent((p2pNode.peers.size / 100) * 100);
        agentsGauge.setPercent((neuralEngine.agents.size / config.neural.maxAgents) * 100);
        // Update DAG chart
        const dagData = {
            title: 'Vertices',
            x: Array.from({ length: 60 }, (_, i) => i.toString()),
            y: Array.from({ length: 60 }, () => Math.random() * 100)
        };
        dagChart.setData([dagData]);
        screen.render();
    };
    // Log function
    const log = (message) => {
        logBox.log(`[${new Date().toISOString()}] ${message}`);
    };
    // Initial log entries
    log('Node started successfully');
    log(`Listening on port ${config.network.port}`);
    log('Waiting for peer connections...');
    // Key bindings
    screen.key(['escape', 'q', 'C-c'], () => {
        screen.destroy();
        process.exit(0);
    });
    // Update UI periodically
    setInterval(updateUI, 1000);
    updateUI();
    // Focus and render
    agentsTable.focus();
    screen.render();
}
async function startMetricsServer(port) {
    const server = (0, http_1.createServer)((req, res) => {
        if (req.url === '/metrics') {
            // Prometheus-style metrics
            const metrics = `
# HELP synaptic_node_info Node information
# TYPE synaptic_node_info gauge
synaptic_node_info{version="1.0.0"} 1

# HELP synaptic_peers_total Total number of connected peers
# TYPE synaptic_peers_total gauge
synaptic_peers_total 0

# HELP synaptic_agents_active Number of active neural agents
# TYPE synaptic_agents_active gauge
synaptic_agents_active 0

# HELP synaptic_dag_vertices_total Total number of DAG vertices
# TYPE synaptic_dag_vertices_total counter
synaptic_dag_vertices_total 0
`;
            res.writeHead(200, { 'Content-Type': 'text/plain' });
            res.end(metrics);
        }
        else {
            res.writeHead(404);
            res.end('Not found');
        }
    });
    server.listen(port, () => {
        console.log(chalk_1.default.cyan(`📊 Metrics server running on port ${port}`));
    });
}
async function shutdownNode(p2pNode, neuralEngine, dagNode) {
    // Graceful shutdown
    await p2pNode.stop();
    // Terminate all agents
    for (const agentId of neuralEngine.agents.keys()) {
        await neuralEngine.terminateAgent(agentId);
    }
    console.log(chalk_1.default.green('✅ Node shutdown complete'));
}
//# sourceMappingURL=start.js.map