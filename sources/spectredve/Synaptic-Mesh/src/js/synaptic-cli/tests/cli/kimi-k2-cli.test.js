/**
 * Kimi-K2 CLI Integration Tests
 * Comprehensive testing for Kimi-K2 integration within Synaptic Neural Mesh CLI
 */

const { describe, test, expect, beforeEach, afterEach } = require('@jest/globals');
const { execSync, spawn } = require('child_process');
const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

describe('Kimi-K2 CLI Integration', () => {
  const testConfigDir = `/tmp/synaptic-test-${uuidv4()}`;
  const configPath = path.join(testConfigDir, 'config.json');
  
  beforeEach(async () => {
    await fs.ensureDir(testConfigDir);
    
    // Create test configuration
    const testConfig = {
      kimi: {
        provider: 'mocktest',
        api_key: 'test-key-encrypted',
        model: 'kimi-k2-instruct',
        temperature: 0.6,
        max_tokens: 4096,
        timeout: 120,
        context_window: 128000
      },
      mesh: {
        node_id: 'test-node-' + uuidv4(),
        port: 18080
      }
    };
    
    await fs.writeJSON(configPath, testConfig);
  });
  
  afterEach(async () => {
    await fs.remove(testConfigDir);
  });

  describe('Configuration Management', () => {
    test('should configure Kimi-K2 API settings', () => {
      const command = `npx synaptic-mesh kimi configure --api-key test-key-123 --provider moonshot --config ${configPath}`;
      
      expect(() => {
        execSync(command, { stdio: 'pipe' });
      }).not.toThrow();
      
      const config = fs.readJSONSync(configPath);
      expect(config.kimi.api_key).toBeDefined();
      expect(config.kimi.provider).toBe('moonshot');
    });

    test('should validate API key format', () => {
      const command = `npx synaptic-mesh kimi configure --api-key invalid-key --config ${configPath}`;
      
      expect(() => {
        execSync(command, { stdio: 'pipe' });
      }).toThrow();
    });

    test('should support multiple providers', () => {
      const providers = ['moonshot', 'openrouter', 'local'];
      
      providers.forEach(provider => {
        const command = `npx synaptic-mesh kimi configure --provider ${provider} --config ${configPath}`;
        
        expect(() => {
          execSync(command, { stdio: 'pipe' });
        }).not.toThrow();
      });
    });
  });

  describe('Query Execution', () => {
    test('should execute basic query', async () => {
      const query = "What is the structure of a neural network?";
      const command = `npx synaptic-mesh kimi query "${query}" --config ${configPath}`;
      
      const result = execSync(command, { 
        encoding: 'utf8',
        timeout: 30000 
      });
      
      expect(result).toContain('neural');
      expect(result.length).toBeGreaterThan(100);
    });

    test('should handle large context queries', async () => {
      const largeQuery = "Analyze this codebase: " + "console.log('test'); ".repeat(1000);
      const command = `npx synaptic-mesh kimi query "${largeQuery}" --config ${configPath}`;
      
      const result = execSync(command, { 
        encoding: 'utf8',
        timeout: 60000 
      });
      
      expect(result).toBeDefined();
    });

    test('should support streaming responses', (done) => {
      const command = `npx synaptic-mesh kimi query "Count from 1 to 10" --stream --config ${configPath}`;
      const child = spawn('npx', command.split(' ').slice(1));
      
      let output = '';
      child.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      child.on('close', (code) => {
        expect(code).toBe(0);
        expect(output).toContain('1');
        expect(output).toContain('10');
        done();
      });
    });
  });

  describe('Tool Integration', () => {
    test('should list available tools', () => {
      const command = `npx synaptic-mesh kimi tools --list --config ${configPath}`;
      const result = execSync(command, { encoding: 'utf8' });
      
      expect(result).toContain('file_operations');
      expect(result).toContain('shell_commands');
      expect(result).toContain('dag_operations');
      expect(result).toContain('neural_operations');
    });

    test('should enable specific tools', () => {
      const command = `npx synaptic-mesh kimi tools --enable file_operations,shell_commands --config ${configPath}`;
      
      expect(() => {
        execSync(command, { stdio: 'pipe' });
      }).not.toThrow();
    });

    test('should execute autonomous tasks with tools', async () => {
      const command = `npx synaptic-mesh kimi execute --autonomous "Create a simple hello.txt file" --config ${configPath}`;
      
      const result = execSync(command, { 
        encoding: 'utf8',
        timeout: 45000 
      });
      
      expect(result).toContain('completed');
      expect(fs.existsSync('hello.txt')).toBe(true);
      
      // Cleanup
      fs.removeSync('hello.txt');
    });
  });

  describe('Status and Monitoring', () => {
    test('should show Kimi-K2 status', () => {
      const command = `npx synaptic-mesh kimi status --config ${configPath}`;
      const result = execSync(command, { encoding: 'utf8' });
      
      expect(result).toContain('provider');
      expect(result).toContain('model');
      expect(result).toContain('context_window');
    });

    test('should show API health check', () => {
      const command = `npx synaptic-mesh kimi health --config ${configPath}`;
      const result = execSync(command, { encoding: 'utf8' });
      
      expect(result).toContain('status');
      expect(result).toMatch(/healthy|degraded|unhealthy/);
    });

    test('should show usage statistics', () => {
      const command = `npx synaptic-mesh kimi stats --config ${configPath}`;
      const result = execSync(command, { encoding: 'utf8' });
      
      expect(result).toContain('requests');
      expect(result).toContain('tokens');
      expect(result).toContain('latency');
    });
  });

  describe('Local Deployment', () => {
    test('should validate hardware requirements', () => {
      const command = `npx synaptic-mesh kimi deploy --validate --engine vllm --config ${configPath}`;
      const result = execSync(command, { encoding: 'utf8' });
      
      expect(result).toContain('gpu_memory');
      expect(result).toContain('system_memory');
      expect(result).toContain('storage_space');
    });

    test('should generate deployment configuration', () => {
      const command = `npx synaptic-mesh kimi deploy --generate-config --engine ktransformers --config ${configPath}`;
      
      expect(() => {
        execSync(command, { stdio: 'pipe' });
      }).not.toThrow();
      
      expect(fs.existsSync('kimi-deployment-config.json')).toBe(true);
      fs.removeSync('kimi-deployment-config.json');
    });

    test('should support different inference engines', () => {
      const engines = ['vllm', 'sglang', 'ktransformers', 'tensorrt-llm'];
      
      engines.forEach(engine => {
        const command = `npx synaptic-mesh kimi deploy --validate --engine ${engine} --config ${configPath}`;
        
        expect(() => {
          execSync(command, { stdio: 'pipe' });
        }).not.toThrow();
      });
    });
  });

  describe('Error Handling', () => {
    test('should handle invalid API keys gracefully', () => {
      const invalidConfig = { ...JSON.parse(fs.readFileSync(configPath)), kimi: { api_key: 'invalid' } };
      const invalidConfigPath = path.join(testConfigDir, 'invalid-config.json');
      fs.writeJSONSync(invalidConfigPath, invalidConfig);
      
      const command = `npx synaptic-mesh kimi query "test" --config ${invalidConfigPath}`;
      
      expect(() => {
        execSync(command, { stdio: 'pipe' });
      }).toThrow(/authentication/i);
    });

    test('should handle network timeouts', () => {
      const timeoutConfig = { ...JSON.parse(fs.readFileSync(configPath)), kimi: { timeout: 1 } };
      const timeoutConfigPath = path.join(testConfigDir, 'timeout-config.json');
      fs.writeJSONSync(timeoutConfigPath, timeoutConfig);
      
      const command = `npx synaptic-mesh kimi query "test" --config ${timeoutConfigPath}`;
      
      expect(() => {
        execSync(command, { stdio: 'pipe', timeout: 5000 });
      }).toThrow(/timeout/i);
    });

    test('should handle rate limiting', () => {
      const commands = Array(10).fill().map(() => 
        `npx synaptic-mesh kimi query "test ${Math.random()}" --config ${configPath}`
      );
      
      // Simulate rapid requests
      expect(() => {
        commands.forEach(cmd => execSync(cmd, { stdio: 'pipe' }));
      }).not.toThrow(); // Should handle rate limiting gracefully
    });
  });

  describe('Security', () => {
    test('should encrypt API keys in configuration', () => {
      const command = `npx synaptic-mesh kimi configure --api-key sk-test-key-123 --config ${configPath}`;
      execSync(command, { stdio: 'pipe' });
      
      const config = fs.readJSONSync(configPath);
      expect(config.kimi.api_key).not.toBe('sk-test-key-123');
      expect(config.kimi.api_key).toMatch(/^encrypted:/);
    });

    test('should validate tool permissions', () => {
      const command = `npx synaptic-mesh kimi execute --autonomous "rm -rf /" --config ${configPath}`;
      
      expect(() => {
        execSync(command, { stdio: 'pipe' });
      }).toThrow(/permission denied|unauthorized/i);
    });

    test('should sandbox tool execution', () => {
      const command = `npx synaptic-mesh kimi execute --autonomous "cat /etc/passwd" --config ${configPath}`;
      
      expect(() => {
        execSync(command, { stdio: 'pipe' });
      }).toThrow(/sandbox|restricted/i);
    });
  });
});

describe('Kimi-K2 Performance Tests', () => {
  const testConfigDir = `/tmp/synaptic-perf-${uuidv4()}`;
  const configPath = path.join(testConfigDir, 'config.json');
  
  beforeEach(async () => {
    await fs.ensureDir(testConfigDir);
    const config = {
      kimi: {
        provider: 'mocktest',
        model: 'kimi-k2-instruct',
        context_window: 128000
      }
    };
    await fs.writeJSON(configPath, config);
  });
  
  afterEach(async () => {
    await fs.remove(testConfigDir);
  });

  test('should handle 128k context window', async () => {
    const largeContext = "word ".repeat(32000); // ~128k tokens
    const command = `npx synaptic-mesh kimi query "Summarize: ${largeContext}" --config ${configPath}`;
    
    const startTime = Date.now();
    const result = execSync(command, { 
      encoding: 'utf8',
      timeout: 120000 
    });
    const endTime = Date.now();
    
    expect(result).toBeDefined();
    expect(endTime - startTime).toBeLessThan(60000); // < 60 seconds
  });

  test('should maintain performance under concurrent load', async () => {
    const concurrentQueries = Array(5).fill().map((_, i) => 
      `npx synaptic-mesh kimi query "Test query ${i}" --config ${configPath}`
    );
    
    const startTime = Date.now();
    const promises = concurrentQueries.map(cmd => 
      new Promise((resolve, reject) => {
        try {
          const result = execSync(cmd, { encoding: 'utf8', timeout: 30000 });
          resolve(result);
        } catch (error) {
          reject(error);
        }
      })
    );
    
    const results = await Promise.all(promises);
    const endTime = Date.now();
    
    expect(results).toHaveLength(5);
    expect(endTime - startTime).toBeLessThan(15000); // < 15 seconds for 5 concurrent
  });

  test('should optimize memory usage', () => {
    const command = `npx synaptic-mesh kimi query "Test memory efficiency" --memory-profile --config ${configPath}`;
    const result = execSync(command, { encoding: 'utf8' });
    
    const memoryMatch = result.match(/memory_usage: (\d+)MB/);
    expect(memoryMatch).toBeTruthy();
    
    const memoryUsage = parseInt(memoryMatch[1]);
    expect(memoryUsage).toBeLessThan(1024); // < 1GB
  });
});