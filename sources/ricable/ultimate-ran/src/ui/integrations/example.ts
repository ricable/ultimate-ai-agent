/**
 * AI Integration Example for TITAN Dashboard
 * Demonstrates how to use Claude Agent SDK and Google ADK for RAN optimization
 *
 * @module ui/integrations/example
 * @version 7.0.0-alpha.1
 */

import { TitanDashboard } from '../titan-dashboard.js';
import { AIOrchestrator } from './ai-orchestrator.js';
import { ClaudeAgentIntegration } from './claude-agent-integration.js';
import { GoogleADKIntegration } from './google-adk-integration.js';
import type { CellStatus, InterferenceMatrix } from '../types.js';

// ============================================================================
// Mock Data for Testing
// ============================================================================

function generateMockCells(count: number = 5): CellStatus[] {
  return Array.from({ length: count }, (_, i) => ({
    cell_id: `Cell-${String(i).padStart(3, '0')}`,
    pci: 100 + i,
    earfcn: 6300,
    power_dbm: -5 + Math.random() * 10,
    tilt_deg: 3 + Math.random() * 8,
    azimuth_deg: (i * 120) % 360,
    latitude: 37.7749 + (Math.random() - 0.5) * 0.1,
    longitude: -122.4194 + (Math.random() - 0.5) * 0.1,
    status: i === 2 ? 'degraded' : 'active' as any,
    kpi: {
      rsrp_avg: -85 + Math.random() * 30,
      rsrq_avg: -10 + Math.random() * 5,
      sinr_avg: i === 2 ? 5 : 10 + Math.random() * 15, // Cell-002 has low SINR
      throughput_mbps: i === 2 ? 30 : 50 + Math.random() * 100,
      rrc_connections: Math.floor(Math.random() * 500),
      prb_utilization: 30 + Math.random() * 50,
      ho_success_rate: 95 + Math.random() * 4,
      drop_rate: Math.random() * 2
    },
    last_optimized: new Date(Date.now() - Math.random() * 86400000).toISOString()
  }));
}

function generateInterferenceMatrix(cells: CellStatus[]): InterferenceMatrix {
  const n = cells.length;
  const matrix: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      i === j ? 0 : -130 + Math.random() * 40
    )
  );

  return {
    cells: cells.map(c => c.cell_id),
    matrix,
    threshold: -90,
    timestamp: new Date().toISOString()
  };
}

// ============================================================================
// Example 1: Using Claude Agent SDK Directly
// ============================================================================

async function example1_ClaudeOnly() {
  console.log('\n='.repeat(80));
  console.log('EXAMPLE 1: Claude Agent SDK - Direct Usage (Opus 4.5)');
  console.log('='.repeat(80));

  // Claude integration now supports both API key and subscription modes
  const model = process.env.ANTHROPIC_MODEL || 'claude-opus-4-5-20251101';
  console.log(`📊 Using model: ${model}`);

  const claude = new ClaudeAgentIntegration({ model });
  const cells = generateMockCells(5);

  console.log(`\n📊 Analyzing ${cells.length} cells with Claude...\n`);

  // Request optimization
  const result = await claude.requestOptimization(
    cells,
    'Improve SINR for degraded cells while minimizing interference',
    { min_power: -10, max_power: 10 }
  );

  console.log('✅ Optimization Results:');
  console.log(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);
  console.log(`   Recommendations: ${result.recommendations.length} parameter changes`);
  console.log('\n📝 Reasoning:');
  console.log(result.reasoning);

  console.log('\n🔧 Recommended Changes:');
  result.recommendations.forEach(rec => {
    console.log(`   ${rec.cell_id}: ${rec.parameter} ${rec.old_value} → ${rec.new_value}`);
  });

  // Analyze specific degraded cell
  const degradedCell = cells.find(c => c.status === 'degraded');
  if (degradedCell) {
    console.log(`\n🔍 Analyzing degraded cell: ${degradedCell.cell_id}\n`);
    const analysis = await claude.analyzeCellPerformance(degradedCell);
    console.log(`   Priority: ${analysis.priority.toUpperCase()}`);
    console.log(`   Issues: ${analysis.issues.length}`);
    analysis.issues.forEach(issue => console.log(`     - ${issue}`));
    console.log(`   Recommendations:`);
    analysis.recommendations.forEach(rec => console.log(`     - ${rec}`));
  }
}

// ============================================================================
// Example 2: Using Google ADK Directly
// ============================================================================

async function example2_GeminiOnly() {
  console.log('\n='.repeat(80));
  console.log('EXAMPLE 2: Google Generative AI (Gemini 3 Pro) - Direct Usage');
  console.log('='.repeat(80));

  // Check for API key (required for Google SDK)
  const apiKey = process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.log('⚠️  GOOGLE_AI_API_KEY not set. Set it with:');
    console.log('   export GOOGLE_AI_API_KEY="your-api-key-here"');
    console.log('   Get key from: https://aistudio.google.com/app/apikey');
    return;
  }

  const model = process.env.GOOGLE_AI_MODEL || 'gemini-2.0-flash-exp';
  console.log(`📊 Using model: ${model}`);

  const gemini = new GoogleADKIntegration({ apiKey, model });
  const cells = generateMockCells(5);
  const interferenceMatrix = generateInterferenceMatrix(cells);

  console.log(`\n📊 Analyzing ${cells.length} cells with Gemini...\n`);

  // Analyze network performance
  const analysis = await gemini.analyzeNetworkPerformance(cells, interferenceMatrix);

  console.log('✅ Network Analysis:');
  console.log(`   Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
  console.log(`   Recommendations: ${analysis.recommendations.length}`);
  console.log('\n📝 Analysis:');
  console.log(analysis.analysis);

  if (analysis.visualInsights && analysis.visualInsights.length > 0) {
    console.log('\n💡 Key Insights:');
    analysis.visualInsights.forEach(insight => console.log(`   - ${insight}`));
  }

  // Detect anomalies
  console.log('\n🚨 Detecting Anomalies...\n');
  const anomalies = await gemini.detectAnomalies(cells);
  console.log(`   Found ${anomalies.anomalies.length} anomalies`);
  anomalies.anomalies.forEach(anomaly => {
    console.log(`   - ${anomaly.cellId}: ${anomaly.metric} (${anomaly.severity})`);
    console.log(`     ${anomaly.description}`);
  });

  // Generate optimization strategy
  console.log('\n📋 Generating Optimization Strategy...\n');
  const strategy = await gemini.generateOptimizationStrategy(
    'Maximize network throughput while maintaining QoS',
    cells
  );
  console.log(`   Timeline: ${strategy.timeline}`);
  console.log(`   Steps: ${strategy.steps.length}`);
  strategy.steps.forEach(step => {
    console.log(`   ${step.step}. ${step.action}`);
  });

  if (strategy.risks.length > 0) {
    console.log('\n⚠️  Risks:');
    strategy.risks.forEach(risk => console.log(`   - ${risk}`));
  }
}

// ============================================================================
// Example 3: Using AI Orchestrator (Hybrid Approach)
// ============================================================================

async function example3_AIOrchestrator() {
  console.log('\n='.repeat(80));
  console.log('EXAMPLE 3: AI Council - Claude Opus 4.5 + Gemini 3 Pro');
  console.log('='.repeat(80));

  // Check for Gemini API key (required)
  const geminiKey = process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY;

  if (!geminiKey) {
    console.log('⚠️  Gemini API key required:');
    console.log('   export GOOGLE_AI_API_KEY="your-gemini-key"');
    console.log('   Get key from: https://aistudio.google.com/app/apikey');
    return;
  }

  // Get models from environment
  const claudeModel = process.env.ANTHROPIC_MODEL || 'claude-opus-4-5-20251101';
  const geminiModel = process.env.GOOGLE_AI_MODEL || 'gemini-2.0-flash-exp';
  const strategy = (process.env.AGENTIC_FLOW_STRATEGY as any) || 'consensus';

  console.log(`📊 Claude Model: ${claudeModel}`);
  console.log(`📊 Gemini Model: ${geminiModel}`);
  console.log(`📊 Strategy: ${strategy}`);

  const orchestrator = new AIOrchestrator({
    claude: { model: claudeModel },
    gemini: { apiKey: geminiKey, model: geminiModel },
    strategy
  });

  const cells = generateMockCells(5);
  const interferenceMatrix = generateInterferenceMatrix(cells);

  console.log(`\n📊 Optimizing ${cells.length} cells with AI Orchestrator (consensus mode)...\n`);

  // Request optimization with consensus
  const result = await orchestrator.requestOptimization(
    cells,
    'Improve SINR and reduce interference',
    interferenceMatrix
  );

  console.log(`✅ Optimization Results (${result.source}):`);
  console.log(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);
  console.log(`   Recommendations: ${result.recommendations.length} consensus changes`);
  console.log('\n📝 Combined Reasoning:');
  console.log(result.reasoning);

  if (result.recommendations.length > 0) {
    console.log('\n🔧 Consensus Recommendations:');
    result.recommendations.forEach(rec => {
      console.log(`   ${rec.cell_id}: ${rec.parameter} ${rec.old_value.toFixed(1)} → ${rec.new_value.toFixed(1)}`);
    });
  }

  // Get network insights
  console.log('\n🌐 Generating Network Insights...\n');
  const insights = await orchestrator.getNetworkInsights(cells, interferenceMatrix);

  console.log('📌 Key Findings:');
  insights.keyFindings.forEach(finding => console.log(`   - ${finding}`));

  if (insights.recommendations.length > 0) {
    console.log('\n💡 Recommendations:');
    insights.recommendations.forEach(rec => console.log(`   - ${rec}`));
  }

  if (insights.risks.length > 0) {
    console.log('\n⚠️  Risks:');
    insights.risks.forEach(risk => console.log(`   - ${risk}`));
  }

  // Analyze specific cell with both AIs
  const targetCell = cells.find(c => c.status === 'degraded') || cells[0];
  console.log(`\n🔍 Deep Analysis of ${targetCell.cell_id} (both AIs)...\n`);

  const cellAnalysis = await orchestrator.analyzeCellPerformance(targetCell);
  console.log(`   Priority: ${cellAnalysis.priority.toUpperCase()}`);
  console.log('\n   Combined Insights:');
  cellAnalysis.combinedInsights.forEach(insight => console.log(`     - ${insight}`));
}

// ============================================================================
// Example 4: Integration with TITAN Dashboard
// ============================================================================

async function example4_DashboardIntegration() {
  console.log('\n='.repeat(80));
  console.log('EXAMPLE 4: Full TITAN Dashboard + AI Council');
  console.log('='.repeat(80));

  // Check for Gemini API key (required)
  const geminiKey = process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY;

  if (!geminiKey) {
    console.log('⚠️  Gemini API key required');
    console.log('   export GOOGLE_AI_API_KEY="your-key"');
    return;
  }

  // Get models from environment
  const claudeModel = process.env.ANTHROPIC_MODEL || 'claude-opus-4-5-20251101';
  const geminiModel = process.env.GOOGLE_AI_MODEL || 'gemini-2.0-flash-exp';
  const strategy = (process.env.AGENTIC_FLOW_STRATEGY as any) || 'parallel';

  console.log(`📊 Council: ${claudeModel} + ${geminiModel} (${strategy})`);

  // Initialize dashboard
  const dashboard = new TitanDashboard({
    port: 8080,
    aguiPort: 3000,
    enableAGUI: false, // Disable AG-UI for this example
    enableOpenTUI: false
  });

  // Initialize AI orchestrator
  const ai = new AIOrchestrator({
    claude: { model: claudeModel },
    gemini: { apiKey: geminiKey, model: geminiModel },
    strategy
  });

  console.log('\n🚀 Starting TITAN Dashboard...\n');

  // Generate mock network
  const cells = generateMockCells(10);
  const interferenceMatrix = generateInterferenceMatrix(cells);

  // Populate dashboard
  cells.forEach(cell => dashboard.updateCellStatus(cell));
  dashboard.renderInterferenceHeatmap(cells, interferenceMatrix, -90);

  console.log(`✅ Dashboard populated with ${cells.length} cells\n`);

  // Use AI to analyze and optimize
  console.log('🤖 Running AI-powered optimization...\n');

  const optimization = await ai.requestOptimization(
    cells.filter(c => c.status === 'degraded' || c.kpi.sinr_avg < 10),
    'Improve SINR for degraded and low-performance cells'
  );

  console.log(`   ${optimization.recommendations.length} optimizations recommended`);

  // Create optimization events in dashboard
  for (const rec of optimization.recommendations) {
    dashboard.addOptimizationEvent({
      id: `ai_opt_${Date.now()}_${rec.cell_id}`,
      timestamp: new Date().toISOString(),
      event_type: 'gnn_decision',
      cell_ids: [rec.cell_id],
      parameters_changed: [rec],
      reasoning: optimization.reasoning.substring(0, 200) + '...',
      confidence: optimization.confidence,
      status: 'pending'
    });
  }

  console.log('   ✅ Events added to dashboard timeline');

  // Create approval request for high-risk changes
  const highRiskChanges = optimization.recommendations.filter(
    rec => Math.abs(rec.new_value - rec.old_value) > 5
  );

  if (highRiskChanges.length > 0) {
    console.log(`\n⚠️  ${highRiskChanges.length} high-risk changes detected. Creating approval request...\n`);

    const approval = dashboard.createApprovalRequest({
      action: 'AI-recommended parameter optimization',
      target: highRiskChanges.map(c => c.cell_id),
      changes: highRiskChanges,
      riskLevel: 'medium',
      justification: optimization.reasoning
    });

    console.log(`   Approval Request: ${approval.id}`);
    console.log(`   Risk Level: ${approval.risk_level}`);

    // Validate with AI
    const validation = await ai.validateApprovalRequest(approval);
    console.log(`\n   AI Validation:`);
    console.log(`     Claude: ${validation.claudeDecision.approved ? '✅ APPROVED' : '❌ REJECTED'}`);
    console.log(`     Gemini: ${validation.geminiDecision.approved ? '✅ APPROVED' : '❌ REJECTED'}`);
    console.log(`     Final: ${validation.finalDecision ? '✅ APPROVED' : '❌ REJECTED'}`);
  }

  console.log('\n✅ Dashboard integration complete!');
  console.log('   View at: http://localhost:8080');
}

// ============================================================================
// Main Runner
// ============================================================================

async function runExamples() {
  console.log('\n╔═══════════════════════════════════════════════════════════════════════════╗');
  console.log('║  TITAN RAN Dashboard - AI Integration Examples                           ║');
  console.log('║  Claude Agent SDK + Google Generative AI (Gemini)                        ║');
  console.log('╚═══════════════════════════════════════════════════════════════════════════╝');

  const examples = [
    { name: 'Claude Only', fn: example1_ClaudeOnly },
    { name: 'Gemini Only', fn: example2_GeminiOnly },
    { name: 'AI Orchestrator', fn: example3_AIOrchestrator },
    { name: 'Dashboard Integration', fn: example4_DashboardIntegration }
  ];

  // Run selected example (or all)
  const exampleIndex = parseInt(process.env.EXAMPLE || '0') - 1;

  if (exampleIndex >= 0 && exampleIndex < examples.length) {
    await examples[exampleIndex].fn();
  } else {
    // Run all examples
    for (const example of examples) {
      try {
        await example.fn();
      } catch (error: any) {
        console.error(`\n❌ Error in ${example.name}:`, error.message);
      }
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('Examples complete!');
  console.log('\nTo run specific examples:');
  console.log('  EXAMPLE=1 npm run ui:integration  # Claude only');
  console.log('  EXAMPLE=2 npm run ui:integration  # Gemini only');
  console.log('  EXAMPLE=3 npm run ui:integration  # AI Orchestrator');
  console.log('  EXAMPLE=4 npm run ui:integration  # Dashboard Integration');
  console.log('='.repeat(80) + '\n');
}

// ============================================================================
// Export
// ============================================================================

export { runExamples };

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runExamples().catch(error => {
    console.error('\n❌ Fatal error:', error);
    process.exit(1);
  });
}
