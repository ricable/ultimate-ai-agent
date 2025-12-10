#!/usr/bin/env node

const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { spawn } = require('child_process');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

async function testMCPDatabaseUpdates() {
    console.log('🧪 Testing MCP Database Updates\n');
    
    // Get initial DB state
    const dbPath = path.join(__dirname, 'data', 'ruv-swarm.db');
    const db = new sqlite3.Database(dbPath);
    
    const getCount = (table) => new Promise((resolve, reject) => {
        db.get(`SELECT COUNT(*) as count FROM ${table}`, (err, row) => {
            if (err) reject(err);
            else resolve(row.count);
        });
    });
    
    // Get initial counts
    const initialCounts = {
        swarms: await getCount('swarms'),
        agents: await getCount('agents'),
        tasks: await getCount('tasks'),
        agent_memory: await getCount('agent_memory')
    };
    
    console.log('📊 Initial Database State:');
    console.log(JSON.stringify(initialCounts, null, 2));
    
    // Start MCP server
    console.log('\n🚀 Starting MCP server...');
    const serverProcess = spawn('node', ['bin/ruv-swarm-clean.js', 'mcp', 'start'], {
        stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // Create MCP client
    const transport = new StdioClientTransport({
        command: 'node',
        args: ['bin/ruv-swarm-clean.js', 'mcp', 'start']
    });
    
    const client = new Client({
        name: 'test-client',
        version: '1.0.0'
    }, {
        capabilities: {}
    });
    
    try {
        await client.connect(transport);
        console.log('✅ Connected to MCP server');
        
        // Initialize a swarm
        console.log('\n📝 Creating new swarm...');
        await client.callTool('swarm_init', {
            topology: 'mesh',
            maxAgents: 3
        });
        
        // Spawn some agents
        console.log('🤖 Spawning agents...');
        await client.callTool('agent_spawn', { type: 'researcher' });
        await client.callTool('agent_spawn', { type: 'coder' });
        
        // Create a task
        console.log('📋 Creating task...');
        await client.callTool('task_orchestrate', {
            task: 'Test database persistence'
        });
        
        // Store memory
        console.log('💾 Storing agent memory...');
        await client.callTool('memory_usage', {
            action: 'store',
            key: 'test/verification',
            value: { timestamp: Date.now(), test: true }
        });
        
        // Wait a bit for DB writes
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Get final counts
        const finalCounts = {
            swarms: await getCount('swarms'),
            agents: await getCount('agents'),
            tasks: await getCount('tasks'),
            agent_memory: await getCount('agent_memory')
        };
        
        console.log('\n📊 Final Database State:');
        console.log(JSON.stringify(finalCounts, null, 2));
        
        console.log('\n📈 Changes:');
        console.log(`Swarms: ${initialCounts.swarms} → ${finalCounts.swarms} (+${finalCounts.swarms - initialCounts.swarms})`);
        console.log(`Agents: ${initialCounts.agents} → ${finalCounts.agents} (+${finalCounts.agents - initialCounts.agents})`);
        console.log(`Tasks: ${initialCounts.tasks} → ${finalCounts.tasks} (+${finalCounts.tasks - initialCounts.tasks})`);
        console.log(`Memory: ${initialCounts.agent_memory} → ${finalCounts.agent_memory} (+${finalCounts.agent_memory - initialCounts.agent_memory})`);
        
        // Check specific entries
        console.log('\n🔍 Verifying specific entries:');
        
        db.all('SELECT * FROM swarms ORDER BY created_at DESC LIMIT 1', (err, rows) => {
            if (!err && rows.length > 0) {
                console.log('✅ Latest swarm:', rows[0]);
            }
        });
        
        db.all('SELECT * FROM agent_memory WHERE key LIKE "test/%" LIMIT 1', (err, rows) => {
            if (!err && rows.length > 0) {
                console.log('✅ Test memory found:', rows[0]);
            }
        });
        
    } catch (error) {
        console.error('❌ Error:', error);
    } finally {
        await client.close();
        serverProcess.kill();
        db.close();
    }
}

// Simpler direct test
async function directDatabaseTest() {
    console.log('\n\n🔧 Direct Database Operation Test\n');
    
    const dbPath = path.join(__dirname, 'data', 'ruv-swarm.db');
    const { DatabaseManager } = require('./src/database');
    const dbManager = new DatabaseManager(dbPath);
    
    // Create a swarm directly
    const swarmId = await dbManager.createSwarm({
        topology: 'test-topology',
        max_agents: 5,
        strategy: 'test-strategy'
    });
    console.log('✅ Created swarm:', swarmId);
    
    // Create an agent
    const agentId = await dbManager.createAgent(swarmId, {
        type: 'test-agent',
        name: 'Test Agent',
        capabilities: ['testing', 'validation']
    });
    console.log('✅ Created agent:', agentId);
    
    // Store memory
    await dbManager.storeMemory(agentId, 'test/direct', {
        timestamp: Date.now(),
        message: 'Direct database test'
    });
    console.log('✅ Stored memory');
    
    // Verify
    const memory = await dbManager.getMemory(agentId, 'test/direct');
    console.log('✅ Retrieved memory:', memory);
}

// Run tests
(async () => {
    try {
        await testMCPDatabaseUpdates();
        await directDatabaseTest();
        console.log('\n✅ All database tests completed successfully!');
    } catch (error) {
        console.error('❌ Test failed:', error);
        process.exit(1);
    }
})();