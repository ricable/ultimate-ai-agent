/**
 * Debug Streaming Example
 *
 * Demonstrates how to use the debug streaming system to get
 * detailed visibility into agent operations.
 */

import { DebugLevel, createDebugStream } from '../src/federation/debug/debug-stream.js';
import { SupabaseFederationAdapterDebug } from '../src/federation/integrations/supabase-adapter-debug.js';

/**
 * Example 1: Basic Debug Streaming
 */
async function basicDebugExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 1: Basic Debug Streaming');
  console.log('='.repeat(60) + '\n');

  // Create debug stream with BASIC level
  const debug = createDebugStream({
    level: DebugLevel.BASIC,
    format: 'human',
    colorize: true,
  });

  // Simulate some operations
  debug.logConnection('server_start', { port: 8443 });

  await new Promise(resolve => setTimeout(resolve, 100));

  debug.logConnection('client_connected', { clientId: 'agent-001' }, 100);

  debug.logDatabase('query_executed', { table: 'agent_sessions', rows: 5 }, 25);

  console.log('\n✅ Basic debug example complete\n');
}

/**
 * Example 2: Detailed Debug with Performance Metrics
 */
async function detailedDebugExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 2: Detailed Debug with Performance Metrics');
  console.log('='.repeat(60) + '\n');

  // Create debug stream with DETAILED level
  const debug = createDebugStream({
    level: DebugLevel.DETAILED,
    format: 'human',
    colorize: true,
    includeTimestamps: true,
    includeMetadata: true,
  });

  // Simulate database operations
  const operations = [
    { op: 'insert_memory', table: 'agent_memories', duration: 15 },
    { op: 'query_memories', table: 'agent_memories', duration: 8 },
    { op: 'update_session', table: 'agent_sessions', duration: 12 },
    { op: 'insert_memory', table: 'agent_memories', duration: 18 },
    { op: 'query_memories', table: 'agent_memories', duration: 7 },
  ];

  for (const { op, table, duration } of operations) {
    debug.logDatabase(op, { table }, duration);
    await new Promise(resolve => setTimeout(resolve, 50));
  }

  console.log('\n📊 Performance Metrics:');
  debug.printMetrics();

  console.log('✅ Detailed debug example complete\n');
}

/**
 * Example 3: Verbose Debug with Full Context
 */
async function verboseDebugExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 3: Verbose Debug with Full Context');
  console.log('='.repeat(60) + '\n');

  // Create debug stream with VERBOSE level
  const debug = createDebugStream({
    level: DebugLevel.VERBOSE,
    format: 'human',
    colorize: true,
    includeTimestamps: true,
    includeMetadata: true,
  });

  // Simulate agent workflow
  debug.logRealtime('agent_join', 'agent-001', {
    tenant: 'team-alpha',
    status: 'online',
  });

  debug.logTask('task_assigned', 'agent-001', 'team-alpha', {
    taskId: 'task-123',
    description: 'Process user request',
    priority: 'high',
  });

  await new Promise(resolve => setTimeout(resolve, 100));

  debug.logMemory('memory_stored', 'agent-001', 'team-alpha', {
    id: 'mem-456',
    content: 'Request processed successfully',
  }, 15);

  debug.logTask('task_completed', 'agent-001', 'team-alpha', {
    taskId: 'task-123',
    result: { status: 'success', items: 42 },
  }, 150);

  debug.logRealtime('broadcast_message', 'agent-001', {
    type: 'status_update',
    message: 'Task complete',
  }, 20);

  console.log('\n✅ Verbose debug example complete\n');
}

/**
 * Example 4: Trace Level Debug
 */
async function traceDebugExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 4: Trace Level Debug (Maximum Detail)');
  console.log('='.repeat(60) + '\n');

  // Create debug stream with TRACE level
  const debug = createDebugStream({
    level: DebugLevel.TRACE,
    format: 'human',
    colorize: true,
    includeTimestamps: true,
    includeStackTraces: false, // Set to true for full stack traces
  });

  // Simulate internal state changes
  debug.logTrace('state_change', {
    component: 'ConnectionPool',
    old_state: 'idle',
    new_state: 'connecting',
  });

  debug.logTrace('config_loaded', {
    config: {
      maxConnections: 10,
      timeout: 5000,
    },
  });

  debug.logDatabase('connection_acquired', { poolId: 'pool-1' }, 5);

  debug.logTrace('state_change', {
    component: 'ConnectionPool',
    old_state: 'connecting',
    new_state: 'connected',
  });

  debug.logDatabase('query_start', {
    sql: 'SELECT * FROM agent_sessions WHERE status = $1',
    params: ['active'],
  });

  await new Promise(resolve => setTimeout(resolve, 50));

  debug.logDatabase('query_complete', {
    rows: 3,
  }, 47);

  debug.logTrace('state_change', {
    component: 'ConnectionPool',
    old_state: 'connected',
    new_state: 'idle',
  });

  console.log('\n✅ Trace debug example complete\n');
}

/**
 * Example 5: Debug with File Output
 */
async function fileDebugExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 5: Debug with File Output');
  console.log('='.repeat(60) + '\n');

  const outputFile = '/tmp/debug-output.log';

  // Create debug stream with file output
  const debug = createDebugStream({
    level: DebugLevel.DETAILED,
    format: 'json', // JSON format for easy parsing
    output: 'both', // Console + file
    outputFile,
    colorize: false, // No color for file output
  });

  debug.logConnection('example_start', { file: outputFile });

  for (let i = 0; i < 5; i++) {
    debug.logDatabase(`operation_${i}`, { iteration: i }, Math.random() * 20);
    await new Promise(resolve => setTimeout(resolve, 50));
  }

  debug.logConnection('example_complete', { file: outputFile });

  // Close file
  debug.close();

  console.log(`\n✅ Debug output written to: ${outputFile}`);
  console.log(`   View with: cat ${outputFile}\n`);
}

/**
 * Example 6: Compact Format for Production
 */
async function compactDebugExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 6: Compact Format (Production-Friendly)');
  console.log('='.repeat(60) + '\n');

  // Create debug stream with compact format
  const debug = createDebugStream({
    level: DebugLevel.DETAILED,
    format: 'compact',
    colorize: false,
  });

  debug.logConnection('server_start', { port: 8443 });
  debug.logDatabase('query', { table: 'sessions' }, 15);
  debug.logRealtime('broadcast', 'agent-001', { msg: 'hello' }, 20);
  debug.logMemory('store', 'agent-001', 'team-1', { id: 'mem-1' }, 12);
  debug.logTask('complete', 'agent-001', 'team-1', { task: 't-1' }, 150);

  console.log('\n✅ Compact debug example complete\n');
}

/**
 * Example 7: Event Filtering
 */
async function filteringDebugExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 7: Event Filtering');
  console.log('='.repeat(60) + '\n');

  // Filter to only show database events
  const debug = createDebugStream({
    level: DebugLevel.VERBOSE,
    format: 'human',
    colorize: true,
    filterCategories: ['database'], // Only show database events
  });

  console.log('Logging various events (only database events will show):\n');

  debug.logConnection('server_start', { port: 8443 }); // Won't show
  debug.logDatabase('query', { table: 'sessions' }, 15); // Will show
  debug.logRealtime('broadcast', 'agent-001', {}); // Won't show
  debug.logDatabase('insert', { table: 'memories' }, 12); // Will show
  debug.logTask('complete', 'agent-001', 'team-1', {}); // Won't show

  console.log('\n✅ Filtering example complete\n');
}

/**
 * Example 8: Real-Time Event Streaming
 */
async function realtimeStreamingExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 8: Real-Time Event Streaming');
  console.log('='.repeat(60) + '\n');

  const debug = createDebugStream({
    level: DebugLevel.VERBOSE,
    format: 'human',
    colorize: true,
  });

  // Subscribe to events
  debug.on('event', (event) => {
    // Custom processing of debug events
    if (event.category === 'database' && event.duration && event.duration > 15) {
      console.log(`\n⚠️  SLOW QUERY DETECTED: ${event.operation} took ${event.duration}ms\n`);
    }
  });

  console.log('Simulating operations with slow query detection:\n');

  debug.logDatabase('fast_query', {}, 8);
  debug.logDatabase('slow_query', {}, 25); // Will trigger alert
  debug.logDatabase('normal_query', {}, 12);
  debug.logDatabase('very_slow_query', {}, 50); // Will trigger alert

  console.log('\n✅ Real-time streaming example complete\n');
}

/**
 * Example 9: Integration with Supabase Adapter (Mock Mode)
 */
async function supabaseAdapterExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 9: Supabase Adapter with Debug Streaming');
  console.log('='.repeat(60) + '\n');

  // This would normally use real Supabase credentials
  console.log('Note: This example shows the structure. Use real credentials for live testing.\n');

  console.log('Example configuration:');
  console.log('  DEBUG_LEVEL=VERBOSE');
  console.log('  DEBUG_FORMAT=human');
  console.log('  DEBUG_OUTPUT=console');
  console.log('  SUPABASE_URL=https://your-project.supabase.co');
  console.log('  SUPABASE_ANON_KEY=your-key\n');

  console.log('With debug enabled, you would see:');
  console.log('  - Connection establishment');
  console.log('  - Table checks');
  console.log('  - Every query with timing');
  console.log('  - Memory operations');
  console.log('  - Performance metrics\n');

  console.log('✅ Supabase adapter example (informational)\n');
}

/**
 * Run all examples
 */
async function main() {
  console.log('\n🔍 Debug Streaming Examples\n');
  console.log('Demonstrating different debug levels and formats:');
  console.log('');

  try {
    await basicDebugExample();
    await detailedDebugExample();
    await verboseDebugExample();
    await traceDebugExample();
    await fileDebugExample();
    await compactDebugExample();
    await filteringDebugExample();
    await realtimeStreamingExample();
    await supabaseAdapterExample();

    console.log('\n' + '='.repeat(60));
    console.log('✅ All Examples Complete!');
    console.log('='.repeat(60));
    console.log('');
    console.log('Debug Levels Available:');
    console.log('  SILENT   (0) - No output');
    console.log('  BASIC    (1) - Major events only');
    console.log('  DETAILED (2) - Include operations');
    console.log('  VERBOSE  (3) - All events');
    console.log('  TRACE    (4) - Everything + internals');
    console.log('');
    console.log('Environment Variables:');
    console.log('  DEBUG_LEVEL=VERBOSE        # Set debug level');
    console.log('  DEBUG_FORMAT=human         # human | json | compact');
    console.log('  DEBUG_OUTPUT=console       # console | file | both');
    console.log('  DEBUG_OUTPUT_FILE=debug.log  # File path');
    console.log('');
  } catch (error) {
    console.error('❌ Error running examples:', error);
    process.exit(1);
  }
}

// Run examples
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
