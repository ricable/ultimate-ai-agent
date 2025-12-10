/**
 * Single Agent Debug Example
 *
 * Demonstrates comprehensive tracking of a single agent's
 * lifecycle, tasks, decisions, and communications.
 */

import { createAgentDebugStream, DebugLevel } from '../src/federation/debug/agent-debug-stream.js';

/**
 * Example: Simulated Agent Lifecycle
 */
async function simulateAgentLifecycle() {
  console.log('\n🤖 Simulating Agent Lifecycle with Debug Streaming\n');

  // Create debug stream for this agent
  const agentDebug = createAgentDebugStream({
    agentId: 'research-agent-001',
    tenantId: 'acme-corp',
    level: DebugLevel.VERBOSE,
    format: 'human',
    colorize: true,
    trackDecisions: true,
    trackCommunication: true,
    timeline: true,
  });

  // Subscribe to events
  agentDebug.on('phase_change', (data) => {
    console.log(`\n📍 Phase Change: ${data.from} → ${data.to}\n`);
  });

  agentDebug.on('decision', (decision) => {
    console.log(`\n🤔 Decision Made: ${decision.context} → ${JSON.stringify(decision.selected)}\n`);
  });

  // Phase 1: Initialization
  console.log('═══ Phase 1: Initialization ═══\n');
  await new Promise(resolve => setTimeout(resolve, 100));

  agentDebug.logInitialization({
    type: 'research-agent',
    capabilities: ['web-search', 'document-analysis', 'summarization'],
    model: 'claude-sonnet-4',
  });

  await new Promise(resolve => setTimeout(resolve, 50));

  agentDebug.logReady({
    capabilities: ['research', 'analyze', 'summarize'],
    status: 'ready',
  });

  // Phase 2: Task Execution
  console.log('\n═══ Phase 2: Task Execution ═══\n');
  await new Promise(resolve => setTimeout(resolve, 100));

  // Start task
  agentDebug.startTask('research-001', 'Research AI safety frameworks');

  // Step 1: Web search
  await new Promise(resolve => setTimeout(resolve, 50));
  agentDebug.logTaskStep('research-001', 0, 'web_search', {
    query: 'AI safety frameworks 2024',
  });

  agentDebug.logThought('Need to search for recent AI safety research', {
    sources: ['arxiv', 'google-scholar'],
  });

  await new Promise(resolve => setTimeout(resolve, 200));
  agentDebug.completeTaskStep('research-001', 0, 200, {
    results_found: 15,
    sources: ['arxiv', 'scholar', 'papers'],
  });

  // Decision: Which papers to analyze
  agentDebug.logDecision(
    'select_papers_to_analyze',
    [
      { title: 'Paper A', relevance: 0.95 },
      { title: 'Paper B', relevance: 0.87 },
      { title: 'Paper C', relevance: 0.82 },
    ],
    { title: 'Paper A', relevance: 0.95 },
    'Highest relevance score',
    0.95
  );

  // Step 2: Document analysis
  await new Promise(resolve => setTimeout(resolve, 50));
  agentDebug.logTaskStep('research-001', 1, 'analyze_document', {
    document: 'Paper A',
  });

  agentDebug.logThought('Analyzing key concepts from selected paper');

  await new Promise(resolve => setTimeout(resolve, 300));
  agentDebug.completeTaskStep('research-001', 1, 300, {
    key_concepts: ['alignment', 'interpretability', 'robustness'],
    word_count: 8500,
  });

  // Step 3: Memory storage
  await new Promise(resolve => setTimeout(resolve, 20));
  agentDebug.logMemoryOperation('store', {
    id: 'mem-001',
    content: 'AI safety requires alignment, interpretability, and robustness',
    embedding_dims: 1536,
  }, 15);

  // Step 4: Summarization
  await new Promise(resolve => setTimeout(resolve, 50));
  agentDebug.logTaskStep('research-001', 2, 'summarize_findings', {
    sections: ['introduction', 'methodology', 'results', 'conclusion'],
  });

  await new Promise(resolve => setTimeout(resolve, 250));
  agentDebug.completeTaskStep('research-001', 2, 250, {
    summary_length: 500,
    key_points: 5,
  });

  // Complete task
  agentDebug.completeTask('research-001', {
    papers_analyzed: 1,
    key_findings: [
      'AI safety requires multiple approaches',
      'Alignment is fundamental',
      'Interpretability enables verification',
    ],
    confidence: 0.92,
  });

  // Phase 3: Communication
  console.log('\n═══ Phase 3: Communication ═══\n');
  await new Promise(resolve => setTimeout(resolve, 100));

  agentDebug.logCommunication('send', 'coordinator-agent', {
    type: 'task_complete',
    task_id: 'research-001',
    summary: 'Research complete with high confidence',
  });

  await new Promise(resolve => setTimeout(resolve, 50));

  agentDebug.logCommunication('receive', 'coordinator-agent', {
    type: 'acknowledgment',
    message: 'Results received, good work',
  });

  // Phase 4: Second Task (faster)
  console.log('\n═══ Phase 4: Second Task ═══\n');
  await new Promise(resolve => setTimeout(resolve, 100));

  agentDebug.startTask('research-002', 'Quick follow-up research');

  await new Promise(resolve => setTimeout(resolve, 50));
  agentDebug.logTaskStep('research-002', 0, 'memory_search', {
    query: 'previous AI safety research',
  });

  agentDebug.logMemoryOperation('search', {
    query_embedding_dims: 1536,
    results: 3,
  }, 5);

  await new Promise(resolve => setTimeout(resolve, 100));
  agentDebug.completeTaskStep('research-002', 0, 100, {
    relevant_memories: 3,
  });

  agentDebug.completeTask('research-002', {
    reused_knowledge: true,
    much_faster: true,
  });

  // Phase 5: Shutdown
  console.log('\n═══ Phase 5: Shutdown ═══\n');
  await new Promise(resolve => setTimeout(resolve, 100));

  agentDebug.logShutdown('Task completed successfully');

  // Print timeline
  console.log('\n');
  agentDebug.printTimeline();

  agentDebug.close();
}

/**
 * Example: Multi-Agent Collaboration with Individual Tracking
 */
async function multiAgentCollaborationExample() {
  console.log('\n\n🤖🤖🤖 Multi-Agent Collaboration (Each Agent Tracked Individually)\n');

  // Create debug stream for researcher
  const researcher = createAgentDebugStream({
    agentId: 'researcher-001',
    tenantId: 'collab-team',
    level: DebugLevel.VERBOSE,
    format: 'human',
    colorize: true,
  });

  // Create debug stream for analyst
  const analyst = createAgentDebugStream({
    agentId: 'analyst-001',
    tenantId: 'collab-team',
    level: DebugLevel.VERBOSE,
    format: 'human',
    colorize: true,
  });

  // Create debug stream for writer
  const writer = createAgentDebugStream({
    agentId: 'writer-001',
    tenantId: 'collab-team',
    level: DebugLevel.VERBOSE,
    format: 'human',
    colorize: true,
  });

  // All agents initialize
  researcher.logInitialization({ type: 'researcher' });
  researcher.logReady();

  analyst.logInitialization({ type: 'analyst' });
  analyst.logReady();

  writer.logInitialization({ type: 'writer' });
  writer.logReady();

  await new Promise(resolve => setTimeout(resolve, 100));

  // Researcher does research
  console.log('\n📚 Researcher Working...\n');
  researcher.startTask('research-task', 'Gather data on topic');
  await new Promise(resolve => setTimeout(resolve, 200));
  researcher.completeTask('research-task', { data_points: 50 });

  researcher.logCommunication('send', 'analyst-001', {
    type: 'data_ready',
    data_points: 50,
  });

  // Analyst receives and processes
  console.log('\n📊 Analyst Working...\n');
  analyst.logCommunication('receive', 'researcher-001', {
    type: 'data_ready',
  });

  analyst.startTask('analysis-task', 'Analyze research data');
  await new Promise(resolve => setTimeout(resolve, 150));
  analyst.completeTask('analysis-task', { insights: 5 });

  analyst.logCommunication('send', 'writer-001', {
    type: 'analysis_ready',
    insights: 5,
  });

  // Writer receives and writes
  console.log('\n✍️  Writer Working...\n');
  writer.logCommunication('receive', 'analyst-001', {
    type: 'analysis_ready',
  });

  writer.startTask('writing-task', 'Write final report');
  await new Promise(resolve => setTimeout(resolve, 200));
  writer.completeTask('writing-task', { word_count: 2000 });

  // All agents shutdown
  console.log('\n🛑 All Agents Shutting Down...\n');
  researcher.logShutdown('Task complete');
  analyst.logShutdown('Task complete');
  writer.logShutdown('Task complete');

  researcher.close();
  analyst.close();
  writer.close();
}

/**
 * Example: Debug a Failing Agent
 */
async function debugFailingAgentExample() {
  console.log('\n\n❌ Debugging a Failing Agent\n');

  const agent = createAgentDebugStream({
    agentId: 'buggy-agent-001',
    tenantId: 'debug-team',
    level: DebugLevel.TRACE, // Maximum verbosity for debugging
    format: 'human',
    colorize: true,
    trackDecisions: true,
  });

  agent.logInitialization({ type: 'data-processor' });
  agent.logReady();

  await new Promise(resolve => setTimeout(resolve, 50));

  agent.startTask('process-data', 'Process large dataset');

  // Step 1: Load data
  agent.logTaskStep('process-data', 0, 'load_data', {
    file: 'large-dataset.csv',
  });

  agent.logThought('Loading data from file...');

  await new Promise(resolve => setTimeout(resolve, 100));
  agent.completeTaskStep('process-data', 0, 100, {
    rows_loaded: 10000,
  });

  // Step 2: Validation (where the bug is)
  agent.logTaskStep('process-data', 1, 'validate_data', {
    validation_rules: ['not_null', 'type_check', 'range_check'],
  });

  agent.logThought('Validating data against rules...');

  await new Promise(resolve => setTimeout(resolve, 50));

  // Decision that leads to error
  agent.logDecision(
    'choose_validation_strategy',
    ['strict', 'lenient', 'skip'],
    'strict', // This causes the error
    'Better to be strict',
    0.8
  );

  // Simulate error
  const error = new Error('Validation failed: 500 rows have invalid data');

  // Log the failure
  agent.failTask('process-data', error);

  agent.logShutdown('Task failed due to validation error');

  console.log('\n💡 Debug Insight:');
  console.log('   The agent chose "strict" validation which caused failure.');
  console.log('   Consider using "lenient" mode for large datasets.\n');

  agent.close();
}

/**
 * Run all examples
 */
async function main() {
  console.log('🔍 Single Agent Debug Streaming Examples');
  console.log('═'.repeat(60) + '\n');

  try {
    await simulateAgentLifecycle();
    await multiAgentCollaborationExample();
    await debugFailingAgentExample();

    console.log('\n' + '═'.repeat(60));
    console.log('✅ All Examples Complete!');
    console.log('═'.repeat(60));
    console.log('\nAgent Debug Features:');
    console.log('  ✅ Lifecycle tracking (spawn → work → shutdown)');
    console.log('  ✅ Task execution with steps');
    console.log('  ✅ Decision logging');
    console.log('  ✅ Communication tracking');
    console.log('  ✅ Memory operations');
    console.log('  ✅ Thoughts/reasoning');
    console.log('  ✅ Performance metrics');
    console.log('  ✅ Timeline visualization');
    console.log('  ✅ Automatic summary');
    console.log('');
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

// Run examples
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
