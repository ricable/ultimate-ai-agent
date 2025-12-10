#!/usr/bin/env node

import { EnhancedMCPTools } from './src/mcp-tools-enhanced.js';
import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function testMCPDatabasePersistence() {
    console.log('🧪 Testing MCP Database Persistence\n');
    
    // Database path
    const dbPath = join(__dirname, 'data', 'ruv-swarm.db');
    console.log(`📂 Database location: ${dbPath}\n`);
    
    // Open database directly to check counts
    const db = new Database(dbPath);
    
    const getTableCounts = () => {
        const counts = {};
        const tables = ['swarms', 'agents', 'tasks', 'agent_memory'];
        
        for (const table of tables) {
            const row = db.prepare(`SELECT COUNT(*) as count FROM ${table}`).get();
            counts[table] = row.count;
        }
        
        return counts;
    };
    
    // Get initial state
    const initialCounts = getTableCounts();
    console.log('📊 Initial Database State:');
    console.log(JSON.stringify(initialCounts, null, 2));
    
    // Create MCP tools instance
    const mcpTools = new EnhancedMCPTools();
    
    console.log('\n🚀 Testing MCP Tool Operations...\n');
    
    try {
        // Test 1: Initialize a swarm
        console.log('1️⃣ Initializing swarm...');
        const swarmResult = await mcpTools.tools.swarm_init({
            topology: 'hierarchical',
            maxAgents: 6,
            strategy: 'adaptive'
        });
        console.log(`✅ Swarm created: ${swarmResult.id}`);
        
        // Test 2: Spawn agents
        console.log('\n2️⃣ Spawning agents...');
        const agent1 = await mcpTools.tools.agent_spawn({
            type: 'researcher',
            name: 'Research Lead',
            capabilities: ['literature review', 'data analysis']
        });
        console.log(`✅ Agent 1 created: ${agent1.agent.id}`);
        
        const agent2 = await mcpTools.tools.agent_spawn({
            type: 'coder',
            name: 'Senior Developer',
            capabilities: ['architecture', 'implementation']
        });
        console.log(`✅ Agent 2 created: ${agent2.agent.id}`);
        
        // Test 3: Create a task
        console.log('\n3️⃣ Orchestrating task...');
        const taskResult = await mcpTools.tools.task_orchestrate({
            task: 'Verify database persistence functionality',
            priority: 'high',
            strategy: 'parallel'
        });
        console.log(`✅ Task created: ${taskResult.task.id}`);
        
        // Test 4: Store memory
        console.log('\n4️⃣ Testing memory storage...');
        const memoryResult = await mcpTools.tools.memory_usage({
            action: 'store',
            key: 'test/persistence/verification',
            value: {
                timestamp: Date.now(),
                test: 'MCP database persistence test',
                agents: [agent1.agent.id, agent2.agent.id],
                swarm: swarmResult.id,
                verified: true
            }
        });
        console.log(`✅ Memory stored with key: ${memoryResult.key}`);
        
        // Test 5: Retrieve memory
        console.log('\n5️⃣ Retrieving stored memory...');
        const retrievedMemory = await mcpTools.tools.memory_usage({
            action: 'retrieve',
            key: 'test/persistence/verification'
        });
        console.log(`✅ Memory retrieved: ${JSON.stringify(retrievedMemory.value, null, 2)}`);
        
        // Wait a moment for any async DB operations
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Get final state
        const finalCounts = getTableCounts();
        console.log('\n📊 Final Database State:');
        console.log(JSON.stringify(finalCounts, null, 2));
        
        // Calculate changes
        console.log('\n📈 Database Changes:');
        for (const table of Object.keys(initialCounts)) {
            const initial = initialCounts[table];
            const final = finalCounts[table];
            const change = final - initial;
            const emoji = change > 0 ? '✅' : '⚠️';
            console.log(`${emoji} ${table}: ${initial} → ${final} (${change > 0 ? '+' : ''}${change})`);
        }
        
        // Query specific records to verify
        console.log('\n🔍 Verifying Specific Records:');
        
        // Check swarm
        const swarmRow = db.prepare('SELECT * FROM swarms WHERE id = ?').get(swarmResult.id);
        if (swarmRow) {
            console.log(`\n✅ Swarm found in database:`);
            console.log(`  - ID: ${swarmRow.id}`);
            console.log(`  - Topology: ${swarmRow.topology}`);
            console.log(`  - Max Agents: ${swarmRow.max_agents}`);
            console.log(`  - Strategy: ${swarmRow.strategy}`);
        } else {
            console.log(`❌ Swarm ${swarmResult.id} NOT found in database!`);
        }
        
        // Check agents
        const agentRows = db.prepare('SELECT * FROM agents WHERE swarm_id = ?').all(swarmResult.id);
        console.log(`\n✅ Found ${agentRows.length} agents in database:`);
        agentRows.forEach(agent => {
            console.log(`  - ${agent.name} (${agent.type}): ${agent.id}`);
        });
        
        // Check memory
        const memoryRow = db.prepare('SELECT * FROM agent_memory WHERE key = ?').get('test/persistence/verification');
        if (memoryRow) {
            console.log(`\n✅ Memory record found:`);
            console.log(`  - Key: ${memoryRow.key}`);
            console.log(`  - Agent: ${memoryRow.agent_id}`);
            console.log(`  - Value: ${memoryRow.value.substring(0, 100)}...`);
        }
        
        // Test swarm status to verify everything is loaded
        console.log('\n6️⃣ Testing swarm status (verifies persistence)...');
        const statusResult = await mcpTools.tools.swarm_status({ verbose: true });
        console.log(`\n✅ Active swarms: ${statusResult.swarms.length}`);
        statusResult.swarms.forEach(swarm => {
            console.log(`  - ${swarm.id}: ${swarm.agents.length} agents`);
        });
        
    } catch (error) {
        console.error('\n❌ Test failed:', error);
        console.error('Stack:', error.stack);
    } finally {
        db.close();
    }
    
    console.log('\n🎉 MCP Database Persistence Test Complete!');
}

// Run the test
testMCPDatabasePersistence().catch(console.error);