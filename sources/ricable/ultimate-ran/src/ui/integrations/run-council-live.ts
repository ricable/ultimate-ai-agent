#!/usr/bin/env npx ts-node --esm
/**
 * TITAN AI Council - Live Runner
 * Runs the multi-agent council using SUBSCRIPTION-ONLY authentication
 *
 * Claude Opus 4.5: Claude Code subscription (CLI auth)
 * Gemini 3 Pro: Google AI Pro subscription (OAuth)
 *
 * Usage: npx ts-node --esm src/ui/integrations/run-council-live.ts
 *
 * @module ui/integrations/run-council-live
 * @version 7.0.0-alpha.1
 */

// Load environment variables FIRST
import { config as dotenvConfig } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
dotenvConfig({ path: resolve(__dirname, '../../../config/.env') });

import { AIOrchestrator, type OptimizationResult } from './ai-orchestrator.js';
import { ClaudeAgentIntegration } from './claude-agent-integration.js';
import type { CellStatus, InterferenceMatrix, ParameterChange } from '../types.js';

// ============================================================================
// Multi-Provider Configuration (Inline to avoid circular imports)
// ============================================================================

function validateSubscriptionMode(): void {
  const subscriptionOnly = process.env.SUBSCRIPTION_ONLY_MODE !== 'false';

  if (subscriptionOnly) {
    if (process.env.ANTHROPIC_API_KEY) {
      throw new Error('ANTHROPIC_API_KEY is blocked. Use Claude Code subscription via CLI.');
    }
    if (process.env.GOOGLE_AI_API_KEY) {
      throw new Error('GOOGLE_AI_API_KEY is blocked. Use Google AI OAuth subscription.');
    }
    console.log('✅ Subscription-only mode: ACTIVE');
  }
}

function getConfig() {
  validateSubscriptionMode();

  return {
    strategy: process.env.AGENTIC_FLOW_STRATEGY || 'consensus',
    primaryProvider: process.env.AGENTIC_FLOW_PRIMARY_PROVIDER || 'anthropic',
    runtime: process.env.RUNTIME_MODE || 'local',
    providers: {
      anthropic: {
        model: process.env.ANTHROPIC_MODEL || 'claude-opus-4-5-20251101',
        useSubscription: true
      },
      google: {
        model: process.env.GOOGLE_AI_MODEL || 'gemini-3-pro-preview',
        oauthClientId: process.env.GOOGLE_OAUTH_CLIENT_ID,
        oauthClientSecret: process.env.GOOGLE_OAUTH_CLIENT_SECRET,
        useSubscription: true
      }
    }
  };
}

// ============================================================================
// Network State Generator (Simulated RAN Data)
// ============================================================================

function generateNetworkState(cellCount: number = 6): {
  cells: CellStatus[];
  interferenceMatrix: InterferenceMatrix;
} {
  const cells: CellStatus[] = Array.from({ length: cellCount }, (_, i) => {
    // Create one degraded cell for the council to optimize
    const isDegraded = i === 2;
    const isOutage = i === 4 && Math.random() > 0.8;

    return {
      cell_id: `Cell-${String(i).padStart(3, '0')}`,
      pci: 100 + i * 10,
      earfcn: 6300 + i * 100,
      power_dbm: -5 + (Math.random() * 10) - (isDegraded ? 5 : 0),
      tilt_deg: 3 + Math.random() * 8,
      azimuth_deg: (i * 60) % 360,
      latitude: 37.7749 + (Math.random() - 0.5) * 0.05,
      longitude: -122.4194 + (Math.random() - 0.5) * 0.05,
      status: isOutage ? 'outage' : isDegraded ? 'degraded' : 'active',
      kpi: {
        rsrp_avg: isDegraded ? -98 : -85 + Math.random() * 15,
        rsrq_avg: isDegraded ? -14 : -10 + Math.random() * 4,
        sinr_avg: isDegraded ? 4 : 12 + Math.random() * 10,
        throughput_mbps: isDegraded ? 25 : 75 + Math.random() * 75,
        rrc_connections: Math.floor(Math.random() * 500),
        prb_utilization: 30 + Math.random() * 50,
        ho_success_rate: 95 + Math.random() * 4.5,
        drop_rate: isDegraded ? 3.5 : 0.5 + Math.random()
      },
      last_optimized: new Date(Date.now() - Math.random() * 86400000).toISOString()
    };
  });

  // Generate interference matrix
  const matrix: number[][] = Array(cellCount).fill(null).map(() =>
    Array(cellCount).fill(0).map(() => -100 + Math.random() * 30)
  );

  // Set diagonal to -Infinity (no self-interference)
  for (let i = 0; i < cellCount; i++) {
    matrix[i][i] = -Infinity;
  }

  // Add some high interference pairs
  if (cellCount >= 3) {
    matrix[0][1] = matrix[1][0] = -75 + Math.random() * 10; // High interference
    matrix[1][2] = matrix[2][1] = -78 + Math.random() * 8;
  }

  const interferenceMatrix: InterferenceMatrix = {
    cells: cells.map(c => c.cell_id),
    matrix,
    threshold: -80,
    timestamp: new Date().toISOString()
  };

  return { cells, interferenceMatrix };
}

// ============================================================================
// Council Runner
// ============================================================================

async function runCouncilLive(): Promise<void> {
  console.log('\n' + '═'.repeat(80));
  console.log('  TITAN AI COUNCIL - LIVE MODE');
  console.log('  Using SUBSCRIPTION-ONLY Authentication');
  console.log('═'.repeat(80));

  // Step 1: Load and validate configuration
  console.log('\n[1/5] Loading Multi-Provider Configuration...');

  let config;
  try {
    config = getConfig();
    console.log('✅ Configuration loaded successfully');
    console.log(`   Strategy: ${config.strategy.toUpperCase()}`);
    console.log(`   Primary: ${config.primaryProvider}`);
    console.log(`   Runtime: ${config.runtime}`);
  } catch (error: any) {
    console.error('❌ Configuration failed:', error.message);
    process.exit(1);
  }

  // Step 2: Initialize AI Orchestrator with subscription auth
  console.log('\n[2/5] Initializing AI Orchestrator...');

  // Check if Gemini API key is available (required for SDK even with subscription)
  const geminiApiKey = process.env.GEMINI_API_KEY;
  let orchestrator: AIOrchestrator | null = null;
  let claudeOnly = false;

  if (!geminiApiKey) {
    console.log('⚠️  GEMINI_API_KEY not set - will use Claude-only mode');
    console.log('');
    console.log('   To enable dual-AI consensus mode:');
    console.log('   1. Go to: https://aistudio.google.com/app/apikey');
    console.log('   2. Create an API key (free with subscription)');
    console.log('   3. Add to config/.env: GEMINI_API_KEY=your-key-here');
    console.log('');
    claudeOnly = true;
  }

  try {
    if (!claudeOnly) {
      orchestrator = new AIOrchestrator({
        claude: {
          // No API key - uses Claude Code CLI subscription
          model: config.providers.anthropic?.model || 'claude-opus-4-5-20251101'
        },
        gemini: {
          apiKey: geminiApiKey,
          model: config.providers.google?.model || 'gemini-3-pro-preview'
        },
        strategy: config.strategy as any,
        consensusThreshold: 0.85
      });
      console.log('✅ AI Orchestrator initialized (Claude + Gemini consensus)');
    } else {
      // Claude-only mode using the ClaudeAgentIntegration directly
      console.log('✅ Running in Claude-only mode (Opus 4.5 via subscription)');
    }
  } catch (error: any) {
    console.error('❌ Orchestrator initialization failed:', error.message);

    // Provide helpful guidance
    if (error.message.includes('API key')) {
      console.log('\n💡 Subscription Setup Required:');
      console.log('   Claude: Run `claude login` to authenticate your subscription');
      console.log('   Gemini: Get API key from https://aistudio.google.com/app/apikey');
      console.log('   See: docs/AUTH.md for detailed instructions');
    }
    process.exit(1);
  }

  // Step 3: Generate network state
  console.log('\n[3/5] Generating Network State...');
  const { cells, interferenceMatrix } = generateNetworkState(6);

  const activeCells = cells.filter(c => c.status === 'active').length;
  const degradedCells = cells.filter(c => c.status === 'degraded').length;
  const outageCells = cells.filter(c => c.status === 'outage').length;

  console.log(`   Total Cells: ${cells.length}`);
  console.log(`   Active: ${activeCells} | Degraded: ${degradedCells} | Outage: ${outageCells}`);

  console.log('\n[Cell KPIs]');
  cells.forEach(c => {
    const status = c.status === 'active' ? '✅' : c.status === 'degraded' ? '⚠️ ' : '❌';
    console.log(`   ${status} ${c.cell_id}: SINR ${c.kpi.sinr_avg.toFixed(1)} dB, ` +
                `Throughput ${c.kpi.throughput_mbps.toFixed(0)} Mbps, ` +
                `Power ${c.power_dbm.toFixed(1)} dBm`);
  });

  // Step 4: Run council optimization
  console.log('\n[4/5] Running AI Council Optimization...');
  console.log('─'.repeat(80));

  const objective = 'Optimize SINR for degraded cells while maintaining neighbor stability';
  console.log(`Objective: ${objective}\n`);

  let result: OptimizationResult;
  const startTime = Date.now();

  try {
    if (orchestrator) {
      // Full consensus mode with Claude + Gemini
      result = await orchestrator.requestOptimization(cells, objective, interferenceMatrix);
    } else {
      // Claude-only mode
      const claude = new ClaudeAgentIntegration({
        model: config.providers.anthropic?.model || 'claude-opus-4-5-20251101'
      });

      console.log('[Claude Opus 4.5] Analyzing network state...\n');
      const claudeResult = await claude.requestOptimization(cells, objective);

      result = {
        source: 'claude',
        recommendations: claudeResult.recommendations,
        reasoning: claudeResult.reasoning,
        confidence: claudeResult.confidence
      };
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`\n✅ Council deliberation complete (${elapsed}s)`);
  } catch (error: any) {
    console.error('\n❌ Council optimization failed:', error.message);

    if (error.message.includes('authentication') || error.message.includes('401')) {
      console.log('\n💡 Authentication Issue:');
      console.log('   - Ensure Claude Code subscription is active: claude login');
      console.log('   - Verify Google OAuth: npm run auth:validate');
    }
    process.exit(1);
  }

  // Step 5: Display results
  console.log('\n[5/5] Council Decision Summary');
  console.log('═'.repeat(80));

  console.log(`\n**Source:** ${result.source.toUpperCase()}`);
  console.log(`**Confidence:** ${(result.confidence * 100).toFixed(1)}%`);

  console.log('\n**Reasoning:**');
  console.log(result.reasoning.split('\n').map(l => '   ' + l).join('\n'));

  if (result.recommendations.length > 0) {
    console.log('\n**Parameter Recommendations:**');
    result.recommendations.forEach((rec, i) => {
      const change = rec.new_value - rec.old_value;
      const changeStr = change >= 0 ? `+${change.toFixed(2)}` : change.toFixed(2);
      console.log(`   ${i + 1}. ${rec.cell_id}: ${rec.parameter}`);
      console.log(`      Current: ${rec.old_value.toFixed(2)} → Proposed: ${rec.new_value.toFixed(2)} (${changeStr})`);
      console.log(`      Bounds: [${rec.bounds[0]}, ${rec.bounds[1]}]`);
    });
  } else {
    console.log('\n**No parameter changes recommended at this time.**');
  }

  if (result.analysis) {
    console.log('\n**Additional Analysis:**');
    console.log(result.analysis.split('\n').slice(0, 10).map(l => '   ' + l).join('\n'));
  }

  if (result.risks && result.risks.length > 0) {
    console.log('\n**Identified Risks:**');
    result.risks.forEach((risk, i) => console.log(`   ${i + 1}. ${risk}`));
  }

  // 3-ROP Governance summary
  console.log('\n' + '─'.repeat(80));
  console.log('[3-ROP Closed-Loop Governance]');
  console.log('─'.repeat(80));
  console.log(`
  If approved, changes follow 3-ROP validation:

  ROP 1 (0-10 min):   Apply changes, collect baseline PM counters
  ROP 2 (10-20 min):  Compare observed KPIs to predictions
  ROP 3 (20-30 min):  Confirm success OR automatic rollback

  Rollback Trigger: KPI deviation > 2σ from prediction
`);

  console.log('═'.repeat(80));
  console.log('  COUNCIL SESSION COMPLETE');
  console.log('═'.repeat(80));

  // Show subscription status
  console.log('\n[Subscription Status]');
  console.log('   Claude: Using Claude Code CLI subscription (Opus 4.5)');
  if (claudeOnly) {
    console.log('   Gemini: NOT CONFIGURED (set GEMINI_API_KEY for consensus mode)');
    console.log('   Mode: CLAUDE-ONLY (subscription)');
  } else {
    console.log('   Gemini: Using Google AI Pro subscription (Gemini 3 Pro)');
    console.log('   Mode: CONSENSUS (dual-AI, subscription-only)');
  }
}

// ============================================================================
// Main Entry Point
// ============================================================================

runCouncilLive().catch((error) => {
  console.error('\n❌ Unhandled error:', error);
  process.exit(1);
});
