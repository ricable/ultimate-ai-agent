#!/usr/bin/env node
/**
 * Comprehensive AgentDB Validation Suite
 * Tests all capabilities: CLI, MCP, Library API, Database Operations
 */

import { spawn } from 'child_process';
import { existsSync, unlinkSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const testDbPath = '/tmp/agentdb-validation-test.db';

// Test results
const results = {
  passed: 0,
  failed: 0,
  tests: []
};

function log(emoji, message) {
  console.log(`${emoji} ${message}`);
}

function test(name, status, details = '') {
  results.tests.push({ name, status, details });
  if (status === 'PASS') {
    results.passed++;
    log('✅', `${name}: PASS${details ? ' - ' + details : ''}`);
  } else {
    results.failed++;
    log('❌', `${name}: FAIL${details ? ' - ' + details : ''}`);
  }
}

async function runCommand(cmd, args, input = null, timeout = 10000) {
  return new Promise((resolve) => {
    const proc = spawn(cmd, args, {
      stdio: input ? 'pipe' : 'inherit',
      shell: true
    });

    let stdout = '';
    let stderr = '';

    if (proc.stdout) {
      proc.stdout.on('data', (data) => stdout += data.toString());
    }
    if (proc.stderr) {
      proc.stderr.on('data', (data) => stderr += data.toString());
    }

    if (input) {
      proc.stdin.write(input);
      proc.stdin.end();
    }

    const timer = setTimeout(() => {
      proc.kill();
      resolve({ code: -1, stdout, stderr, timeout: true });
    }, timeout);

    proc.on('close', (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
    });
  });
}

async function testCLI() {
  log('🧪', '\n=== Testing CLI Commands ===\n');

  // Clean up test database
  if (existsSync(testDbPath)) unlinkSync(testDbPath);

  // Test 1: Version
  let result = await runCommand('node', ['dist/cli/agentdb-cli.js', '--version']);
  test('CLI: --version', result.stdout.includes('agentdb v') ? 'PASS' : 'FAIL',
       result.stdout.trim());

  // Test 2: Help
  result = await runCommand('node', ['dist/cli/agentdb-cli.js', 'help']);
  test('CLI: help', result.stdout.includes('AgentDB CLI') ? 'PASS' : 'FAIL');

  // Test 3: Init database
  result = await runCommand('node', ['dist/cli/agentdb-cli.js', 'init', testDbPath]);
  const dbExists = existsSync(testDbPath);
  test('CLI: init database', dbExists ? 'PASS' : 'FAIL',
       dbExists ? `${testDbPath} created` : 'File not created');

  // Test 4: Database stats
  if (dbExists) {
    result = await runCommand('node', ['dist/cli/agentdb-cli.js', 'db', 'stats']);
    test('CLI: db stats', result.code === 0 ? 'PASS' : 'FAIL');
  }
}

async function testMCP() {
  log('🧪', '\n=== Testing MCP Server ===\n');

  // Test 1: MCP server starts
  const mcpRequest = JSON.stringify({
    jsonrpc: '2.0',
    method: 'tools/list',
    id: 1
  }) + '\n';

  const result = await runCommand('node', ['dist/mcp/agentdb-mcp-server.js'], mcpRequest, 5000);

  const hasStartupMessage = result.stderr.includes('AgentDB MCP Server');
  test('MCP: Server starts', hasStartupMessage ? 'PASS' : 'FAIL');

  const hasToolsList = result.stdout.includes('"tools":');
  test('MCP: Returns tools list', hasToolsList ? 'PASS' : 'FAIL');

  if (hasToolsList) {
    try {
      const response = JSON.parse(result.stdout);
      const toolCount = response.result?.tools?.length || 0;
      test('MCP: Tool count', toolCount >= 25 ? 'PASS' : 'FAIL', `${toolCount} tools`);
    } catch (e) {
      test('MCP: JSON response', 'FAIL', 'Invalid JSON');
    }
  }
}

async function testLibraryAPI() {
  log('🧪', '\n=== Testing Library API ===\n');

  try {
    // Test 1: Import main module
    const agentdb = await import('../dist/index.js');
    test('Library: Import main module', agentdb ? 'PASS' : 'FAIL');

    // Test 2: Check exports
    const expectedExports = [
      'EmbeddingService',
      'ReflexionMemory',
      'SkillLibrary',
      'CausalMemoryGraph',
      'LearningSystem',
      'createDatabase'
    ];

    const exports = Object.keys(agentdb);
    const hasAllExports = expectedExports.every(exp => exports.includes(exp));
    test('Library: Required exports', hasAllExports ? 'PASS' : 'FAIL',
         `${exports.length} exports available`);

    // Test 3: Create EmbeddingService
    const { EmbeddingService } = agentdb;
    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    test('Library: Create EmbeddingService', embedder ? 'PASS' : 'FAIL');

    // Test 4: Initialize embedder
    await embedder.initialize();
    test('Library: Initialize embeddings', true, 'Initialized without error');

    // Test 5: Generate embedding
    const embedding = await embedder.embed('test');
    test('Library: Generate embedding',
         embedding.length === 384 ? 'PASS' : 'FAIL',
         `${embedding.length} dimensions`);

    // Test 6: Create database
    const { createDatabase } = agentdb;
    if (existsSync(testDbPath)) unlinkSync(testDbPath);
    const db = await createDatabase(testDbPath);
    test('Library: Create database', db ? 'PASS' : 'FAIL');

    // Test 7: Database operations
    db.exec(`CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, data TEXT)`);
    db.prepare('INSERT INTO test (data) VALUES (?)').run('test data');
    const row = db.prepare('SELECT * FROM test WHERE id = ?').get(1);
    test('Library: Database operations', row?.data === 'test data' ? 'PASS' : 'FAIL');

    db.close();

  } catch (error) {
    test('Library: API test', 'FAIL', error.message);
  }
}

async function testDatabaseSchema() {
  log('🧪', '\n=== Testing Database Schema ===\n');

  try {
    const agentdb = await import('../dist/index.js');
    const { createDatabase } = agentdb;

    if (existsSync(testDbPath)) unlinkSync(testDbPath);

    // Initialize with schema
    const db = await createDatabase(testDbPath);
    const schemaPath = join(__dirname, '../dist/schemas/schema.sql');
    const schema = readFileSync(schemaPath, 'utf-8');
    db.exec(schema);

    // Check for expected tables
    const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all();
    const tableNames = tables.map(t => t.name);

    const expectedTables = [
      'episodes',
      'embeddings',
      'skills',
      'skill_embeddings',
      'causal_edges',
      'reflexion_memories'
    ];

    const hasAllTables = expectedTables.every(t => tableNames.includes(t));
    test('Schema: Expected tables', hasAllTables ? 'PASS' : 'FAIL',
         `${tableNames.length} tables found`);

    // Test episode insertion
    const insertResult = db.prepare(`
      INSERT INTO episodes (text, embedding_vector, metadata, created_at)
      VALUES (?, ?, ?, ?)
    `).run('test', JSON.stringify([0.1, 0.2, 0.3]), '{}', Date.now());

    test('Schema: Insert episode', insertResult.changes === 1 ? 'PASS' : 'FAIL');

    db.close();

  } catch (error) {
    test('Schema: Test', 'FAIL', error.message);
  }
}

async function testSecurity() {
  log('🧪', '\n=== Testing Security Features ===\n');

  try {
    const security = await import('../dist/security/input-validation.js');

    // Test 1: Table name validation
    try {
      security.validateTableName('episodes');
      test('Security: Valid table name', 'PASS');
    } catch (e) {
      test('Security: Valid table name', 'FAIL', e.message);
    }

    // Test 2: Invalid table name
    try {
      security.validateTableName('DROP TABLE users;');
      test('Security: Invalid table name blocked', 'FAIL', 'Should have thrown error');
    } catch (e) {
      test('Security: Invalid table name blocked', 'PASS', 'SQL injection prevented');
    }

    // Test 3: Safe WHERE clause
    const whereClause = security.buildSafeWhereClause({ session_id: 'test123' });
    test('Security: Safe WHERE clause',
         whereClause.includes('session_id = ?') ? 'PASS' : 'FAIL');

  } catch (error) {
    test('Security: Test', 'FAIL', error.message);
  }
}

async function main() {
  console.log('🚀 AgentDB Comprehensive Validation Suite\n');
  console.log('Testing AgentDB v1.4.9 capabilities...\n');

  await testCLI();
  await testMCP();
  await testLibraryAPI();
  await testDatabaseSchema();
  await testSecurity();

  // Summary
  console.log('\n' + '='.repeat(50));
  console.log('📊 VALIDATION SUMMARY');
  console.log('='.repeat(50));
  console.log(`✅ Passed: ${results.passed}`);
  console.log(`❌ Failed: ${results.failed}`);
  console.log(`📈 Success Rate: ${(results.passed / (results.passed + results.failed) * 100).toFixed(1)}%`);
  console.log('='.repeat(50) + '\n');

  // Detailed results
  if (results.failed > 0) {
    console.log('❌ Failed Tests:');
    results.tests.filter(t => t.status === 'FAIL').forEach(t => {
      console.log(`   - ${t.name}: ${t.details}`);
    });
    console.log();
  }

  // Cleanup
  if (existsSync(testDbPath)) unlinkSync(testDbPath);

  process.exit(results.failed > 0 ? 1 : 0);
}

main().catch(console.error);
