/**
 * Demo Script - Ericsson RAN Swarm Optimizer
 *
 * Demonstrates the complete autonomous optimization pipeline:
 * 1. Initialize simulation with realistic RAN environment
 * 2. Create swarm of NanoAgents for each cell
 * 3. Run optimization cycles with federated learning
 * 4. Inject faults and observe self-healing
 * 5. Display real-time statistics
 */

import {
  NanoAgent,
  createSwarm,
  SwarmSimulator,
  CellGlobalIdentity
} from '../src/index.js';

// Configuration
const DEMO_CONFIG = {
  numCells: 7,       // 7-cell cluster (center + 6 neighbors)
  numUsers: 200,
  simulationDuration: 60,  // seconds
  tickIntervalMs: 1000
};

async function runDemo() {
  console.log('\n🚀 Starting Ericsson RAN Swarm Optimizer Demo\n');
  console.log('=' .repeat(60));

  // Step 1: Initialize Simulator
  console.log('\n📡 Initializing RAN Simulator...');
  const simulator = new SwarmSimulator({
    numCells: DEMO_CONFIG.numCells,
    numUsers: DEMO_CONFIG.numUsers,
    areaKm2: 1,
    tickIntervalMs: DEMO_CONFIG.tickIntervalMs,
    scenario: 'urban',
    trafficProfile: 'hotspot',
    faultInjection: {
      enabled: true,
      faultTypes: ['UPLINK_INTERFERENCE', 'CAPACITY_SATURATION'],
      probability: 0.05  // 5% chance per tick
    }
  });

  simulator.initialize();
  const simStats = simulator.getStats();
  console.log(`   ✓ Created ${simStats.cellCount} cells with ${simStats.userCount} users`);

  // Step 2: Create Swarm of NanoAgents
  console.log('\n🤖 Creating NanoAgent Swarm...');
  const cellKpis = simulator.getAllCellKpis();
  const cells: CellGlobalIdentity[] = cellKpis.map(k => k.cgi);

  const agents = createSwarm(cells, 'demo-cluster');
  console.log(`   ✓ Created ${agents.size} autonomous NanoAgents`);

  // Step 3: Wire up event handlers
  console.log('\n🔗 Connecting Event Handlers...');

  let problemsDetected = 0;
  let actionsExecuted = 0;
  let actionsBlocked = 0;

  for (const [agentId, agent] of agents) {
    agent.on('problem-detected', ({ problem }) => {
      problemsDetected++;
      console.log(`\n⚠️  [${agentId}] Problem detected: ${problem.category}`);
      console.log(`   Root cause: ${problem.rootCause || 'analyzing...'}`);
    });

    agent.on('action-executed', ({ action }) => {
      actionsExecuted++;
      console.log(`   ✓ [${agentId}] Action executed: ${action.type}`);
    });

    agent.on('action-blocked', ({ violations }) => {
      actionsBlocked++;
      console.log(`   ✗ [${agentId}] Action blocked: ${violations[0]?.violation || 'security'}`);
    });

    agent.on('federation-round', ({ round }) => {
      console.log(`   🔄 [${agentId}] Federated learning round ${round} complete`);
    });
  }

  // Simulator events
  simulator.on('fault-injected', ({ cellId, faultType }) => {
    console.log(`\n💥 Fault injected: ${faultType} on ${cellId}`);
  });

  simulator.on('fault-cleared', ({ cellId, faultType }) => {
    console.log(`   ✅ Fault cleared: ${faultType} on ${cellId}`);
  });

  console.log('   ✓ Event handlers connected');

  // Step 4: Run Simulation Loop
  console.log('\n▶️  Starting Simulation...\n');
  console.log('=' .repeat(60));

  const startTime = Date.now();
  let tick = 0;

  const runTick = async () => {
    tick++;
    const elapsed = (Date.now() - startTime) / 1000;

    if (elapsed >= DEMO_CONFIG.simulationDuration) {
      return false;
    }

    // Step simulation
    const state = simulator.step();

    // Feed KPIs to agents
    for (const [cellId, kpis] of state.cells) {
      const agent = agents.get(cellId);
      if (agent) {
        agent.processKPI(kpis);
      }
    }

    // Get RAN graph for spatial processing
    const graph = simulator.getRANGraph();

    // Run optimization on each agent periodically
    if (tick % 10 === 0) {
      for (const [cellId, agent] of agents) {
        const cellKpi = state.cells.get(cellId);
        if (cellKpi) {
          const neighborKpis = Array.from(state.cells.values())
            .filter(k => simulator['cellSimulator'].cgiToString(k.cgi) !== cellId);

          agent.updateSpatialContext(graph);
          agent.runOptimization(
            cellKpi,
            neighborKpis,
            {
              p0NominalPusch: -96,
              alpha: 0.8
            }
          );
        }
      }
    }

    // Print status every 10 ticks
    if (tick % 10 === 0) {
      const stats = simulator.getStats();
      process.stdout.write(
        `\r⏱️  Tick ${tick.toString().padStart(3)} | ` +
        `Load: ${stats.avgLoad.toFixed(0)}% | ` +
        `Throughput: ${stats.avgThroughput.toFixed(0)} Mbps | ` +
        `Faults: ${stats.activeFaults} | ` +
        `Problems: ${problemsDetected} | ` +
        `Actions: ${actionsExecuted}/${actionsBlocked} `
      );
    }

    return true;
  };

  // Run simulation loop
  while (await runTick()) {
    await new Promise(r => setTimeout(r, DEMO_CONFIG.tickIntervalMs));
  }

  // Step 5: Print Final Statistics
  console.log('\n\n' + '=' .repeat(60));
  console.log('📊 FINAL STATISTICS\n');

  const finalSimStats = simulator.getStats();
  console.log('Simulation:');
  console.log(`   Total ticks: ${finalSimStats.tick}`);
  console.log(`   Total handovers: ${finalSimStats.totalHandovers}`);
  console.log(`   Average load: ${finalSimStats.avgLoad.toFixed(1)}%`);
  console.log(`   Average throughput: ${finalSimStats.avgThroughput.toFixed(1)} Mbps`);

  console.log('\nSwarm:');
  console.log(`   Agents: ${agents.size}`);
  console.log(`   Problems detected: ${problemsDetected}`);
  console.log(`   Actions executed: ${actionsExecuted}`);
  console.log(`   Actions blocked: ${actionsBlocked}`);

  // Print per-agent stats
  console.log('\nPer-Agent Statistics:');
  for (const [agentId, agent] of agents) {
    const stats = agent.getStats();
    console.log(`\n   Agent ${agentId}:`);
    console.log(`      Trajectories stored: ${stats.reasoningBankStats.trajectoryCount}`);
    console.log(`      GNN forward passes: ${stats.stGnnStats.forwardPasses}`);
    console.log(`      Anomalies detected: ${stats.midstreamerStats.anomaliesDetected}`);
    console.log(`      Optimization actions: ${stats.optimizerStats.actionsExecuted}`);
    console.log(`      Active problems: ${stats.fmStats.activeProblems}`);
    console.log(`      Security blocks: ${stats.securityStats.actionsBlocked}`);
  }

  console.log('\n' + '=' .repeat(60));
  console.log('✅ Demo Complete!\n');
}

// Run the demo
runDemo().catch(console.error);
