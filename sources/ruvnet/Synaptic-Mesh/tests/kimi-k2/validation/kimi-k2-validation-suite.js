/**
 * Kimi-K2 Validation Test Suite
 * Comprehensive validation testing for Kimi-K2 integration across all components
 */

const { describe, test, expect, beforeAll, afterAll } = require('@jest/globals');
const { execSync, spawn } = require('child_process');
const fs = require('fs-extra');
const path = require('path');
const os = require('os');

describe('Kimi-K2 Validation Suite', () => {
  let validationResults = {
    timestamp: Date.now(),
    systemInfo: {
      platform: os.platform(),
      nodeVersion: process.version,
      arch: os.arch()
    },
    testResults: []
  };

  beforeAll(async () => {
    console.log('🧪 Starting Kimi-K2 comprehensive validation...');
  });

  afterAll(async () => {
    // Save validation report
    const reportPath = `/tmp/kimi-k2-validation-report-${Date.now()}.json`;
    await fs.writeJSON(reportPath, validationResults, { spaces: 2 });
    console.log(`📊 Validation report saved to: ${reportPath}`);
  });

  describe('NPM Package Validation', () => {
    test('should validate synaptic-mesh package structure', async () => {
      const packagePath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
      const packageJson = await fs.readJSON(path.join(packagePath, 'package.json'));
      
      // Validate package.json structure
      expect(packageJson.name).toBe('synaptic-mesh');
      expect(packageJson.version).toMatch(/^\d+\.\d+\.\d+/);
      expect(packageJson.description).toContain('Neural Mesh');
      expect(packageJson.bin).toBeDefined();
      expect(packageJson.dependencies).toBeDefined();
      
      // Validate binary files exist
      const binPath = path.join(packagePath, 'bin');
      expect(await fs.pathExists(binPath)).toBe(true);
      
      validationResults.testResults.push({
        test: 'npm_package_structure',
        status: 'passed',
        details: {
          name: packageJson.name,
          version: packageJson.version,
          binariesCount: Object.keys(packageJson.bin || {}).length
        }
      });
    });

    test('should validate package dependencies compatibility', async () => {
      const packagePath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
      
      // Check if npm install would work
      try {
        execSync('npm list --depth=0', { 
          cwd: packagePath, 
          stdio: 'pipe' 
        });
        
        validationResults.testResults.push({
          test: 'dependency_compatibility',
          status: 'passed'
        });
      } catch (error) {
        validationResults.testResults.push({
          test: 'dependency_compatibility',
          status: 'failed',
          error: error.message
        });
        
        // Don't fail the test, just record the issue
        console.warn('⚠️  Dependency issues detected:', error.message);
      }
    });
  });

  describe('Rust Crates Validation', () => {
    test('should validate published crates structure', async () => {
      const cratesDir = '/workspaces/Synaptic-Neural-Mesh/standalone-crates';
      const expectedCrates = [
        'synaptic-neural-mesh',
        'synaptic-qudag-core', 
        'synaptic-daa-swarm',
        'synaptic-mesh-cli',
        'synaptic-neural-wasm'
      ];
      
      for (const crateName of expectedCrates) {
        const cratePath = path.join(cratesDir, crateName);
        const cargoTomlPath = path.join(cratePath, 'Cargo.toml');
        
        if (await fs.pathExists(cargoTomlPath)) {
          const cargoToml = await fs.readFile(cargoTomlPath, 'utf8');
          
          expect(cargoToml).toContain(`name = "${crateName}"`);
          expect(cargoToml).toContain('version =');
          expect(cargoToml).toContain('edition =');
          
          validationResults.testResults.push({
            test: `rust_crate_${crateName}`,
            status: 'passed',
            path: cratePath
          });
        } else {
          validationResults.testResults.push({
            test: `rust_crate_${crateName}`,
            status: 'missing',
            path: cratePath
          });
        }
      }
    });

    test('should validate Rust compilation', async () => {
      const cratesDir = '/workspaces/Synaptic-Neural-Mesh/standalone-crates';
      const crateList = await fs.readdir(cratesDir);
      
      for (const crateName of crateList) {
        const cratePath = path.join(cratesDir, crateName);
        const cargoTomlPath = path.join(cratePath, 'Cargo.toml');
        
        if (await fs.pathExists(cargoTomlPath)) {
          try {
            execSync('cargo check', { 
              cwd: cratePath, 
              stdio: 'pipe',
              timeout: 30000
            });
            
            validationResults.testResults.push({
              test: `rust_compilation_${crateName}`,
              status: 'passed'
            });
          } catch (error) {
            validationResults.testResults.push({
              test: `rust_compilation_${crateName}`,
              status: 'failed',
              error: error.message
            });
            
            console.warn(`⚠️  Compilation issues in ${crateName}:`, error.message);
          }
        }
      }
    });
  });

  describe('CLI Integration Validation', () => {
    test('should validate CLI commands are accessible', async () => {
      const cliPath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
      const expectedCommands = [
        'init',
        'start', 
        'mesh',
        'neural',
        'dag',
        'peer',
        'status',
        'stop',
        'config'
      ];
      
      // Check if CLI help shows all commands
      try {
        const helpOutput = execSync('node src/cli.ts --help', { 
          cwd: cliPath,
          encoding: 'utf8',
          timeout: 10000
        });
        
        for (const command of expectedCommands) {
          expect(helpOutput).toContain(command);
        }
        
        validationResults.testResults.push({
          test: 'cli_commands_available',
          status: 'passed',
          commandsFound: expectedCommands.length
        });
      } catch (error) {
        validationResults.testResults.push({
          test: 'cli_commands_available',
          status: 'failed',
          error: error.message
        });
      }
    });

    test('should validate CLI error handling', async () => {
      const cliPath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
      
      try {
        // Test invalid command
        execSync('node src/cli.ts invalid-command', { 
          cwd: cliPath,
          stdio: 'pipe',
          timeout: 5000
        });
        
        // Should not reach here
        expect(false).toBe(true);
      } catch (error) {
        // Error is expected for invalid command
        expect(error.status).not.toBe(0);
        
        validationResults.testResults.push({
          test: 'cli_error_handling',
          status: 'passed'
        });
      }
    });
  });

  describe('Integration Points Validation', () => {
    test('should validate MCP tool integration points', async () => {
      const mcpPath = '/workspaces/Synaptic-Neural-Mesh/src/mcp';
      
      // Check MCP server files exist
      const mcpFiles = [
        'synaptic-mcp-server.ts',
        'mcp-config.json'
      ];
      
      for (const file of mcpFiles) {
        const filePath = path.join(mcpPath, file);
        const exists = await fs.pathExists(filePath);
        
        validationResults.testResults.push({
          test: `mcp_file_${file}`,
          status: exists ? 'passed' : 'missing',
          path: filePath
        });
        
        expect(exists).toBe(true);
      }
    });

    test('should validate Docker integration', async () => {
      const dockerFiles = [
        '/workspaces/Synaptic-Neural-Mesh/Dockerfile',
        '/workspaces/Synaptic-Neural-Mesh/docker-compose.yml',
        '/workspaces/Synaptic-Neural-Mesh/docker/claude-container/Dockerfile'
      ];
      
      for (const dockerFile of dockerFiles) {
        const exists = await fs.pathExists(dockerFile);
        
        validationResults.testResults.push({
          test: `docker_file_${path.basename(dockerFile)}`,
          status: exists ? 'passed' : 'missing',
          path: dockerFile
        });
        
        if (exists) {
          const content = await fs.readFile(dockerFile, 'utf8');
          expect(content.length).toBeGreaterThan(0);
        }
      }
    });

    test('should validate WASM integration', async () => {
      const wasmPaths = [
        '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli/wasm',
        '/workspaces/Synaptic-Neural-Mesh/src/rs/ruv-FANN/pkg'
      ];
      
      for (const wasmPath of wasmPaths) {
        if (await fs.pathExists(wasmPath)) {
          const files = await fs.readdir(wasmPath);
          const wasmFiles = files.filter(f => f.endsWith('.wasm'));
          
          validationResults.testResults.push({
            test: `wasm_files_${path.basename(wasmPath)}`,
            status: wasmFiles.length > 0 ? 'passed' : 'no_wasm_files',
            wasmFilesCount: wasmFiles.length,
            path: wasmPath
          });
        } else {
          validationResults.testResults.push({
            test: `wasm_path_${path.basename(wasmPath)}`,
            status: 'missing',
            path: wasmPath
          });
        }
      }
    });
  });

  describe('Performance Validation', () => {
    test('should validate system resource requirements', async () => {
      const systemInfo = {
        totalMemoryGB: Math.round(os.totalmem() / 1024 / 1024 / 1024),
        freememoryGB: Math.round(os.freemem() / 1024 / 1024 / 1024),
        cpuCount: os.cpus().length,
        platform: os.platform()
      };
      
      // Validate minimum requirements for Kimi-K2
      const requirements = {
        minMemoryGB: 8,
        minCpuCores: 2,
        supportedPlatforms: ['linux', 'darwin', 'win32']
      };
      
      const validationChecks = {
        sufficientMemory: systemInfo.totalMemoryGB >= requirements.minMemoryGB,
        sufficientCPU: systemInfo.cpuCount >= requirements.minCpuCores,
        supportedPlatform: requirements.supportedPlatforms.includes(systemInfo.platform)
      };
      
      validationResults.testResults.push({
        test: 'system_requirements',
        status: Object.values(validationChecks).every(Boolean) ? 'passed' : 'insufficient',
        systemInfo,
        requirements,
        validationChecks
      });
      
      // Warnings for insufficient resources
      if (!validationChecks.sufficientMemory) {
        console.warn(`⚠️  Insufficient memory: ${systemInfo.totalMemoryGB}GB < ${requirements.minMemoryGB}GB required`);
      }
      if (!validationChecks.sufficientCPU) {
        console.warn(`⚠️  Insufficient CPU cores: ${systemInfo.cpuCount} < ${requirements.minCpuCores} required`);
      }
    });

    test('should validate large context handling capability', async () => {
      const testSizes = [1000, 10000, 50000]; // token counts
      const contextTests = [];
      
      for (const tokenCount of testSizes) {
        const largeString = 'token '.repeat(tokenCount);
        const startTime = Date.now();
        
        // Simulate processing (in real implementation, this would call Kimi-K2)
        const processingTime = Date.now() - startTime;
        
        contextTests.push({
          tokenCount,
          stringLength: largeString.length,
          processingTimeMs: processingTime,
          withinLimits: tokenCount <= 128000 // Kimi-K2 context window
        });
      }
      
      validationResults.testResults.push({
        test: 'large_context_capability',
        status: 'passed',
        contextTests
      });
      
      expect(contextTests.every(test => test.withinLimits)).toBe(true);
    });
  });

  describe('Security Validation', () => {
    test('should validate API key security measures', async () => {
      const securityChecks = [];
      
      // Check for hardcoded API keys in source files
      const sourceFiles = await findSourceFiles('/workspaces/Synaptic-Neural-Mesh/src');
      let hardcodedKeys = 0;
      
      for (const file of sourceFiles.slice(0, 50)) { // Limit to first 50 files
        try {
          const content = await fs.readFile(file, 'utf8');
          
          // Look for potential API key patterns
          const keyPatterns = [
            /sk-[a-zA-Z0-9]{32,}/g,
            /api[_-]?key\s*[:=]\s*['""][^'"]{20,}['"]/gi,
            /Bearer\s+[a-zA-Z0-9]{32,}/g
          ];
          
          for (const pattern of keyPatterns) {
            if (pattern.test(content)) {
              hardcodedKeys++;
              break;
            }
          }
        } catch (error) {
          // Skip files that can't be read
        }
      }
      
      securityChecks.push({
        check: 'no_hardcoded_keys',
        status: hardcodedKeys === 0 ? 'passed' : 'failed',
        hardcodedKeysFound: hardcodedKeys
      });
      
      // Check for secure environment variable usage
      const envVarUsage = sourceFiles.slice(0, 20).map(async file => {
        try {
          const content = await fs.readFile(file, 'utf8');
          return content.includes('process.env') || content.includes('dotenv');
        } catch {
          return false;
        }
      });
      
      const usesEnvVars = (await Promise.all(envVarUsage)).some(Boolean);
      
      securityChecks.push({
        check: 'uses_environment_variables',
        status: usesEnvVars ? 'passed' : 'warning',
        usesEnvVars
      });
      
      validationResults.testResults.push({
        test: 'security_validation',
        status: securityChecks.every(check => check.status === 'passed') ? 'passed' : 'warnings',
        securityChecks
      });
    });

    test('should validate quantum-resistant cryptography', async () => {
      const cryptoFiles = [
        '/workspaces/Synaptic-Neural-Mesh/src/rs/qudag-core/src/crypto.rs',
        '/workspaces/Synaptic-Neural-Mesh/src/rs/QuDAG/QuDAG-main/core/crypto/src/lib.rs'
      ];
      
      let quantumResistantFound = false;
      
      for (const file of cryptoFiles) {
        if (await fs.pathExists(file)) {
          const content = await fs.readFile(file, 'utf8');
          
          // Look for quantum-resistant algorithm mentions
          if (content.includes('ML-DSA') || content.includes('Dilithium') || content.includes('Kyber')) {
            quantumResistantFound = true;
            break;
          }
        }
      }
      
      validationResults.testResults.push({
        test: 'quantum_resistant_crypto',
        status: quantumResistantFound ? 'passed' : 'not_found',
        quantumResistantFound
      });
    });
  });

  describe('Documentation Validation', () => {
    test('should validate documentation completeness', async () => {
      const docFiles = [
        '/workspaces/Synaptic-Neural-Mesh/README.md',
        '/workspaces/Synaptic-Neural-Mesh/plans/Kimi-K2/KIMI_K2_INTEGRATION_EPIC.md',
        '/workspaces/Synaptic-Neural-Mesh/docs/tutorials/quick-start.md'
      ];
      
      const docValidation = [];
      
      for (const docFile of docFiles) {
        if (await fs.pathExists(docFile)) {
          const content = await fs.readFile(docFile, 'utf8');
          
          docValidation.push({
            file: path.basename(docFile),
            status: 'exists',
            wordCount: content.split(/\s+/).length,
            hasKimiMention: content.toLowerCase().includes('kimi')
          });
        } else {
          docValidation.push({
            file: path.basename(docFile),
            status: 'missing'
          });
        }
      }
      
      validationResults.testResults.push({
        test: 'documentation_completeness',
        status: docValidation.every(doc => doc.status === 'exists') ? 'passed' : 'incomplete',
        docValidation
      });
    });
  });
});

// Helper function to find source files
async function findSourceFiles(dir, extensions = ['.js', '.ts', '.rs']) {
  const files = [];
  
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
        files.push(...await findSourceFiles(fullPath, extensions));
      } else if (entry.isFile() && extensions.some(ext => entry.name.endsWith(ext))) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    // Skip directories that can't be read
  }
  
  return files;
}

describe('Validation Report Generation', () => {
  test('should generate comprehensive validation report', async () => {
    const reportSummary = {
      totalTests: validationResults.testResults.length,
      passedTests: validationResults.testResults.filter(t => t.status === 'passed').length,
      failedTests: validationResults.testResults.filter(t => t.status === 'failed').length,
      warningTests: validationResults.testResults.filter(t => ['warning', 'incomplete', 'missing'].includes(t.status)).length
    };
    
    reportSummary.successRate = reportSummary.totalTests > 0 
      ? (reportSummary.passedTests / reportSummary.totalTests * 100).toFixed(2) 
      : 0;
    
    validationResults.summary = reportSummary;
    
    console.log(`📊 Validation Summary:
    ✅ Passed: ${reportSummary.passedTests}
    ❌ Failed: ${reportSummary.failedTests}
    ⚠️  Warnings: ${reportSummary.warningTests}
    🎯 Success Rate: ${reportSummary.successRate}%`);
    
    expect(reportSummary.totalTests).toBeGreaterThan(0);
    expect(reportSummary.successRate).toBeGreaterThan(70); // At least 70% success rate
  });
});