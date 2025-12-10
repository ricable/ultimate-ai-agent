/**
 * Kimi-K2 MCP Tools Integration Tests
 * Testing Model Context Protocol integration with Kimi-K2
 */

const { describe, test, expect, beforeEach, afterEach } = require('@jest/globals');
const { MCPClient } = require('../../lib/mcp-client');
const { KimiK2MCPBridge } = require('../../lib/kimi-k2-mcp-bridge');
const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

describe('Kimi-K2 MCP Tools Integration', () => {
  let mcpClient;
  let kimiMCPBridge;
  let testSessionId;
  
  beforeEach(async () => {
    testSessionId = `test-session-${uuidv4()}`;
    mcpClient = new MCPClient({
      server: 'synaptic-mesh-mcp',
      transport: 'stdio'
    });
    
    kimiMCPBridge = new KimiK2MCPBridge({
      mcpClient,
      model: 'kimi-k2-instruct',
      sessionId: testSessionId
    });
    
    await mcpClient.connect();
  });
  
  afterEach(async () => {
    await mcpClient.disconnect();
    await kimiMCPBridge.cleanup();
  });

  describe('MCP Tool Discovery', () => {
    test('should discover available MCP tools for Kimi-K2', async () => {
      const tools = await kimiMCPBridge.discoverTools();
      
      expect(tools).toBeInstanceOf(Array);
      expect(tools.length).toBeGreaterThan(0);
      
      const expectedTools = [
        'file_operations',
        'shell_commands',
        'dag_operations', 
        'neural_mesh_operations',
        'memory_operations',
        'swarm_coordination'
      ];
      
      expectedTools.forEach(toolName => {
        const tool = tools.find(t => t.name === toolName);
        expect(tool).toBeDefined();
        expect(tool.description).toBeTruthy();
        expect(tool.parameters).toBeDefined();
      });
    });

    test('should validate tool schemas', async () => {
      const tools = await kimiMCPBridge.discoverTools();
      
      tools.forEach(tool => {
        expect(tool.type).toBe('function');
        expect(tool.function).toBeDefined();
        expect(tool.function.name).toBeTruthy();
        expect(tool.function.description).toBeTruthy();
        expect(tool.function.parameters).toBeDefined();
        expect(tool.function.parameters.type).toBe('object');
      });
    });

    test('should support tool filtering by capability', async () => {
      const fileTools = await kimiMCPBridge.discoverTools(['file_operations']);
      const shellTools = await kimiMCPBridge.discoverTools(['shell_commands']);
      
      expect(fileTools.length).toBeGreaterThan(0);
      expect(shellTools.length).toBeGreaterThan(0);
      expect(fileTools).not.toEqual(shellTools);
    });
  });

  describe('Tool Execution via MCP', () => {
    test('should execute file read operations', async () => {
      const testFile = '/tmp/test-kimi-read.txt';
      await fs.writeFile(testFile, 'Test content for Kimi-K2');
      
      const result = await kimiMCPBridge.executeTool('file_operations', {
        operation: 'read',
        path: testFile
      });
      
      expect(result.success).toBe(true);
      expect(result.content).toBe('Test content for Kimi-K2');
      
      await fs.remove(testFile);
    });

    test('should execute file write operations', async () => {
      const testFile = '/tmp/test-kimi-write.txt';
      const content = 'Content written by Kimi-K2 via MCP';
      
      const result = await kimiMCPBridge.executeTool('file_operations', {
        operation: 'write',
        path: testFile,
        content: content
      });
      
      expect(result.success).toBe(true);
      expect(await fs.pathExists(testFile)).toBe(true);
      expect(await fs.readFile(testFile, 'utf8')).toBe(content);
      
      await fs.remove(testFile);
    });

    test('should execute shell commands safely', async () => {
      const result = await kimiMCPBridge.executeTool('shell_commands', {
        command: 'echo "Hello from Kimi-K2"',
        timeout: 5000
      });
      
      expect(result.success).toBe(true);
      expect(result.stdout).toContain('Hello from Kimi-K2');
      expect(result.stderr).toBe('');
      expect(result.exitCode).toBe(0);
    });

    test('should handle DAG operations', async () => {
      const dagResult = await kimiMCPBridge.executeTool('dag_operations', {
        operation: 'query',
        query: 'SELECT * FROM nodes LIMIT 5'
      });
      
      expect(dagResult.success).toBe(true);
      expect(dagResult.nodes).toBeInstanceOf(Array);
    });

    test('should coordinate with neural mesh', async () => {
      const meshResult = await kimiMCPBridge.executeTool('neural_mesh_operations', {
        operation: 'agent_spawn',
        agent_type: 'analyzer',
        capabilities: ['data_analysis', 'pattern_recognition']
      });
      
      expect(meshResult.success).toBe(true);
      expect(meshResult.agent_id).toBeTruthy();
    });
  });

  describe('Autonomous Tool Selection', () => {
    test('should autonomously select appropriate tools for tasks', async () => {
      const task = "Analyze the contents of a JSON file and create a summary report";
      
      const execution = await kimiMCPBridge.executeAutonomousTask(task, {
        maxSteps: 10,
        allowedTools: ['file_operations', 'shell_commands']
      });
      
      expect(execution.success).toBe(true);
      expect(execution.steps).toBeInstanceOf(Array);
      expect(execution.steps.length).toBeGreaterThan(0);
      
      // Check that appropriate tools were selected
      const toolsUsed = execution.steps.map(step => step.tool);
      expect(toolsUsed).toContain('file_operations');
    });

    test('should handle multi-step autonomous workflows', async () => {
      const task = "Create a directory, write a file with system info, and then read it back";
      
      const execution = await kimiMCPBridge.executeAutonomousTask(task, {
        maxSteps: 5,
        allowedTools: ['file_operations', 'shell_commands']
      });
      
      expect(execution.success).toBe(true);
      expect(execution.steps.length).toBeGreaterThanOrEqual(3);
      
      // Verify the sequence of operations
      const operations = execution.steps.map(step => step.parameters?.operation || step.tool);
      expect(operations).toContain('shell_commands'); // mkdir
      expect(operations).toContain('file_operations'); // write and read
    });

    test('should respect tool permissions and security', async () => {
      const maliciousTask = "Delete all files in the system";
      
      const execution = await kimiMCPBridge.executeAutonomousTask(maliciousTask, {
        maxSteps: 5,
        allowedTools: ['file_operations']
      });
      
      expect(execution.success).toBe(false);
      expect(execution.error).toMatch(/permission denied|security violation/i);
    });
  });

  describe('Context and Memory Management', () => {
    test('should maintain context across tool executions', async () => {
      // Execute first tool
      const firstResult = await kimiMCPBridge.executeTool('file_operations', {
        operation: 'write',
        path: '/tmp/context-test.txt',
        content: 'Context test data'
      });
      
      // Execute second tool that should be aware of the first
      const secondResult = await kimiMCPBridge.executeTool('file_operations', {
        operation: 'read',
        path: '/tmp/context-test.txt'
      });
      
      expect(firstResult.success).toBe(true);
      expect(secondResult.success).toBe(true);
      expect(secondResult.content).toBe('Context test data');
      
      // Check context history
      const context = await kimiMCPBridge.getExecutionContext();
      expect(context.history).toHaveLength(2);
      expect(context.history[0].tool).toBe('file_operations');
      expect(context.history[1].tool).toBe('file_operations');
      
      await fs.remove('/tmp/context-test.txt');
    });

    test('should handle large context windows efficiently', async () => {
      const largeData = 'x'.repeat(50000); // 50KB of data
      
      const result = await kimiMCPBridge.executeTool('memory_operations', {
        operation: 'store',
        key: 'large_context_test',
        value: largeData
      });
      
      expect(result.success).toBe(true);
      
      const retrieved = await kimiMCPBridge.executeTool('memory_operations', {
        operation: 'retrieve',
        key: 'large_context_test'
      });
      
      expect(retrieved.success).toBe(true);
      expect(retrieved.value).toBe(largeData);
    });

    test('should persist tool execution history', async () => {
      await kimiMCPBridge.executeTool('shell_commands', {
        command: 'echo "persistence test"'
      });
      
      const history = await kimiMCPBridge.getExecutionHistory();
      expect(history.length).toBeGreaterThan(0);
      expect(history[history.length - 1].tool).toBe('shell_commands');
      expect(history[history.length - 1].timestamp).toBeDefined();
    });
  });

  describe('Error Handling and Recovery', () => {
    test('should handle tool execution errors gracefully', async () => {
      const result = await kimiMCPBridge.executeTool('shell_commands', {
        command: 'nonexistent-command'
      });
      
      expect(result.success).toBe(false);
      expect(result.error).toBeTruthy();
      expect(result.stderr).toBeTruthy();
    });

    test('should retry failed operations with backoff', async () => {
      const unstableOperation = jest.fn()
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Timeout'))
        .mockResolvedValueOnce({ success: true });
      
      kimiMCPBridge._executeToolOperation = unstableOperation;
      
      const result = await kimiMCPBridge.executeTool('test_tool', {
        data: 'test'
      });
      
      expect(result.success).toBe(true);
      expect(unstableOperation).toHaveBeenCalledTimes(3);
    });

    test('should handle MCP server disconnections', async () => {
      await mcpClient.disconnect();
      
      const result = await kimiMCPBridge.executeTool('file_operations', {
        operation: 'read',
        path: '/tmp/test.txt'
      });
      
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/connection|server/i);
    });
  });

  describe('Performance Optimization', () => {
    test('should cache tool schema information', async () => {
      const startTime = Date.now();
      await kimiMCPBridge.discoverTools();
      const firstCallTime = Date.now() - startTime;
      
      const startTime2 = Date.now();
      await kimiMCPBridge.discoverTools();
      const secondCallTime = Date.now() - startTime2;
      
      expect(secondCallTime).toBeLessThan(firstCallTime * 0.5); // Should be much faster
    });

    test('should handle concurrent tool executions', async () => {
      const operations = Array(5).fill().map((_, i) => 
        kimiMCPBridge.executeTool('shell_commands', {
          command: `echo "Concurrent test ${i}"`
        })
      );
      
      const results = await Promise.all(operations);
      
      expect(results).toHaveLength(5);
      results.forEach((result, i) => {
        expect(result.success).toBe(true);
        expect(result.stdout).toContain(`Concurrent test ${i}`);
      });
    });

    test('should optimize memory usage for large contexts', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Execute many operations with large data
      for (let i = 0; i < 10; i++) {
        await kimiMCPBridge.executeTool('memory_operations', {
          operation: 'store',
          key: `large_data_${i}`,
          value: 'x'.repeat(10000)
        });
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024; // MB
      
      expect(memoryIncrease).toBeLessThan(100); // Should not use more than 100MB
    });
  });
});

describe('Kimi-K2 MCP Integration Edge Cases', () => {
  let kimiMCPBridge;
  
  beforeEach(async () => {
    kimiMCPBridge = new KimiK2MCPBridge({
      model: 'kimi-k2-instruct',
      sessionId: `edge-case-${uuidv4()}`
    });
  });

  test('should handle malformed tool parameters', async () => {
    const result = await kimiMCPBridge.executeTool('file_operations', {
      invalid_param: 'test',
      missing_required: undefined
    });
    
    expect(result.success).toBe(false);
    expect(result.error).toMatch(/parameter|validation/i);
  });

  test('should handle extremely large tool outputs', async () => {
    const result = await kimiMCPBridge.executeTool('shell_commands', {
      command: 'yes | head -n 100000'
    });
    
    expect(result.success).toBe(true);
    expect(result.stdout.length).toBeGreaterThan(100000);
  });

  test('should handle binary file operations', async () => {
    const binaryData = Buffer.from([0x89, 0x50, 0x4E, 0x47]); // PNG header
    const testFile = '/tmp/test-binary.png';
    
    await fs.writeFile(testFile, binaryData);
    
    const result = await kimiMCPBridge.executeTool('file_operations', {
      operation: 'read',
      path: testFile,
      encoding: 'base64'
    });
    
    expect(result.success).toBe(true);
    expect(result.content).toBe(binaryData.toString('base64'));
    
    await fs.remove(testFile);
  });
});