#!/usr/bin/env npx tsx
/**
 * Network Optimization CLI
 *
 * Usage: npx tsx src/cli/optimize.ts [options]
 *
 * Runs the GNN-based optimization agent swarm on network configurations.
 */

import { promises as fs } from 'fs';
import * as path from 'path';
import { NetworkGraphBuilder } from '../gnn/network-graph.js';
import { GNNInferenceEngine } from '../gnn/gnn-engine.js';
import { SwarmController } from '../agents/agent-swarm.js';
import { logger } from '../utils/logger.js';
import { getConfig } from '../core/config.js';
import type { CellConfiguration, PerformanceMetrics, AlphaValue } from '../core/types.js';

// Parse command line arguments
function parseArgs(): {
  configFile?: string;
  metricsFile?: string;
  outputFile?: string;
  dryRun: boolean;
  verbose: boolean;
} {
  const args = process.argv.slice(2);
  const result = {
    configFile: undefined as string | undefined,
    metricsFile: undefined as string | undefined,
    outputFile: undefined as string | undefined,
    dryRun: false,
    verbose: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--config' || arg === '-c') {
      result.configFile = args[++i];
    } else if (arg === '--metrics' || arg === '-m') {
      result.metricsFile = args[++i];
    } else if (arg === '--output' || arg === '-o') {
      result.outputFile = args[++i];
    } else if (arg === '--dry-run' || arg === '-n') {
      result.dryRun = true;
    } else if (arg === '--verbose' || arg === '-v') {
      result.verbose = true;
    } else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    }
  }

  return result;
}

function printUsage(): void {
  console.log(`
RuVector Network Optimization CLI
==================================

Usage: npx tsx src/cli/optimize.ts [options]

Options:
  -c, --config <file>    Cell configuration JSON file
  -m, --metrics <file>   Performance metrics JSON file
  -o, --output <file>    Output file for recommendations
  -n, --dry-run          Simulate without generating output
  -v, --verbose          Verbose output
  -h, --help             Show this help message

If no files are provided, uses demo data for illustration.

Configuration File Format (JSON):
{
  "cells": [
    {
      "ecgi": "310-260-12345-1",
      "powerControl": {
        "pZeroNominalPusch": -100,
        "alpha": 1.0,
        "pZeroNominalPucch": -100,
        "pCmax": 23
      },
      "antennaTilt": 4,
      "azimuth": 120,
      "height": 30,
      "maxTxPower": 43,
      "bandwidth": 20,
      "band": 7,
      "technology": "LTE"
    }
  ],
  "neighborRelations": {
    "310-260-12345-1": ["310-260-12345-2", "310-260-12345-3"]
  }
}

Example:
  npx tsx src/cli/optimize.ts -c ./data/cells.json -o ./output/recommendations.json
`);
}

// Generate demo data for illustration
function generateDemoData(): {
  cells: CellConfiguration[];
  neighborRelations: Map<string, string[]>;
} {
  const cells: CellConfiguration[] = [];
  const neighborRelations = new Map<string, string[]>();

  // Generate a small cluster of 9 cells (3x3 grid)
  for (let i = 0; i < 9; i++) {
    const cellId = `demo-cell-${i + 1}`;

    cells.push({
      ecgi: cellId,
      powerControl: {
        pZeroNominalPusch: -100 + Math.floor(Math.random() * 10) - 5,
        alpha: 1.0 as AlphaValue, // Default full compensation
        pZeroNominalPucch: -100,
        pCmax: 23,
      },
      antennaTilt: 4 + Math.floor(Math.random() * 4),
      azimuth: (i * 40) % 360,
      height: 25 + Math.floor(Math.random() * 10),
      maxTxPower: 43,
      bandwidth: 20,
      band: 7,
      technology: 'LTE',
    });
  }

  // Create neighbor relations (grid topology)
  for (let i = 0; i < 9; i++) {
    const neighbors: string[] = [];
    const row = Math.floor(i / 3);
    const col = i % 3;

    // Add adjacent cells
    if (col > 0) neighbors.push(`demo-cell-${i}`);
    if (col < 2) neighbors.push(`demo-cell-${i + 2}`);
    if (row > 0) neighbors.push(`demo-cell-${i - 2}`);
    if (row < 2) neighbors.push(`demo-cell-${i + 4}`);

    neighborRelations.set(`demo-cell-${i + 1}`, neighbors);
  }

  return { cells, neighborRelations };
}

async function main() {
  const args = parseArgs();
  const config = getConfig();

  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║           RuVector Network Optimization CLI                      ║
╠══════════════════════════════════════════════════════════════════╣
║  GNN-Based Power Control Optimization for Ericsson RAN           ║
║  Solving the Tuning Paradox with Bayesian Uncertainty            ║
╚══════════════════════════════════════════════════════════════════╝
`);

  let cells: CellConfiguration[];
  let neighborRelations: Map<string, string[]>;

  // Load or generate data
  if (args.configFile) {
    console.log(`Loading configuration from: ${args.configFile}`);
    const configData = JSON.parse(await fs.readFile(args.configFile, 'utf-8'));
    cells = configData.cells;
    neighborRelations = new Map(Object.entries(configData.neighborRelations || {}));
  } else {
    console.log('No configuration file provided, using demo data...\n');
    const demoData = generateDemoData();
    cells = demoData.cells;
    neighborRelations = demoData.neighborRelations;
  }

  console.log(`Cells loaded: ${cells.length}`);
  console.log(`Neighbor relations: ${neighborRelations.size}`);
  console.log('');

  // Build network graph
  console.log('Building network graph...');
  const graphBuilder = new NetworkGraphBuilder();
  const graph = graphBuilder.buildGraph(cells, new Map(), neighborRelations);
  graphBuilder.normalizeFeatures(graph);

  console.log(`Graph created: ${graph.nodes.size} nodes, ${graph.edges.length} edges`);
  console.log('');

  // Analyze current state
  console.log('Current Network Configuration:');
  console.log('┌─────────────────┬─────────┬───────┬───────────┐');
  console.log('│ Cell ID         │ P0 (dBm)│ Alpha │ Technology│');
  console.log('├─────────────────┼─────────┼───────┼───────────┤');

  for (const [cellId, node] of graph.nodes) {
    const p0 = node.config.powerControl.pZeroNominalPusch;
    const alpha = node.config.powerControl.alpha;
    const tech = node.config.technology;
    console.log(
      `│ ${cellId.padEnd(15)} │ ${String(p0).padStart(7)} │ ${String(alpha).padStart(5)} │ ${tech.padEnd(9)} │`
    );
  }
  console.log('└─────────────────┴─────────┴───────┴───────────┘');
  console.log('');

  // Create agent swarm
  console.log('Initializing agent swarm...');
  const swarmController = new SwarmController();
  const swarm = swarmController.createSwarm(graph.clusterId);
  console.log(`  Optimizer: ${swarm.optimizer.name}`);
  console.log(`  Validator: ${swarm.validator.name}`);
  console.log(`  Auditor: ${swarm.auditor.name}`);
  console.log('');

  // Run optimization
  console.log('Running optimization (Genetic Algorithm + GNN Simulation)...');
  console.log('━'.repeat(66));

  const startTime = Date.now();
  const action = await swarmController.runOptimizationCycle(graph, swarm);
  const duration = ((Date.now() - startTime) / 1000).toFixed(2);

  console.log('━'.repeat(66));
  console.log(`Optimization completed in ${duration}s`);
  console.log('');

  if (!action) {
    console.log('Result: No beneficial changes found');
    console.log('The current configuration appears to be near-optimal.');
    return;
  }

  // Display results
  console.log('╔══════════════════════════════════════════════════════════════════╗');
  console.log('║                    OPTIMIZATION RESULTS                          ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║  Status: ${action.status.toUpperCase().padEnd(54)}║`);
  console.log(`║  Action ID: ${action.id.substring(0, 36).padEnd(51)}║`);
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log('║  Predicted Improvements:                                         ║');
  console.log(`║    SINR Improvement: ${(action.prediction.sinrImprovement * 100).toFixed(2).padStart(6)}%                              ║`);
  console.log(`║    Spectral Efficiency Gain: ${(action.prediction.spectralEfficiencyGain * 100).toFixed(2).padStart(6)}%                      ║`);
  console.log(`║    Coverage Impact: ${(action.prediction.coverageImpact * 100).toFixed(2).padStart(7)}%                               ║`);
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log('║  Uncertainty Analysis (Bayesian):                                ║');
  console.log(`║    Total Uncertainty: ${action.prediction.uncertainty.toFixed(4).padStart(10)}                            ║`);
  console.log(`║    Epistemic: ${action.prediction.epistemicUncertainty.toFixed(4).padStart(10)}                                    ║`);
  console.log(`║    Aleatoric: ${action.prediction.aleatoricUncertainty.toFixed(4).padStart(10)}                                    ║`);
  console.log(`║    95% CI: [${action.prediction.confidenceInterval[0].toFixed(4)}, ${action.prediction.confidenceInterval[1].toFixed(4)}]                           ║`);
  console.log('╚══════════════════════════════════════════════════════════════════╝');
  console.log('');

  // Display recommended changes
  console.log('Recommended Parameter Changes:');
  console.log('┌─────────────────┬───────────────┬────────────┬────────────┐');
  console.log('│ Cell ID         │ Parameter     │ Old Value  │ New Value  │');
  console.log('├─────────────────┼───────────────┼────────────┼────────────┤');

  for (const change of action.changes) {
    const cellId = change.cellId.padEnd(15).substring(0, 15);
    const param = change.parameter.substring(0, 13).padEnd(13);
    const oldVal = String(change.oldValue).padStart(10);
    const newVal = String(change.newValue).padStart(10);
    console.log(`│ ${cellId} │ ${param} │ ${oldVal} │ ${newVal} │`);
  }

  console.log('└─────────────────┴───────────────┴────────────┴────────────┘');
  console.log('');

  // Save output if requested
  if (args.outputFile && !args.dryRun) {
    const output = {
      timestamp: new Date().toISOString(),
      graphId: graph.id,
      clusterId: graph.clusterId,
      action: {
        id: action.id,
        status: action.status,
        prediction: action.prediction,
        changes: action.changes,
        targetCells: action.targetCells,
      },
      optimizationDuration: parseFloat(duration),
    };

    const outputPath = path.isAbsolute(args.outputFile)
      ? args.outputFile
      : path.resolve(process.cwd(), args.outputFile);

    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, JSON.stringify(output, null, 2));
    console.log(`Results saved to: ${outputPath}`);
  }

  console.log(`
Note: These are recommendations based on GNN simulation.
The 3-ROP Stability Protocol should be used to validate changes in production.

For questions about specific parameters, use:
  npx tsx src/cli/query.ts "What is the optimal alpha value for urban deployments?"
`);

  logger.info('Optimization completed', {
    graphId: graph.id,
    actionId: action?.id,
    duration: parseFloat(duration),
    changesRecommended: action?.changes.length || 0,
  });
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
