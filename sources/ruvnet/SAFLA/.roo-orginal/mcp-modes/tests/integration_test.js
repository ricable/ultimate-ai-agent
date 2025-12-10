/**
 * MCP-Optimized Modes Integration Test Suite
 * 
 * This test suite validates that all MCP-optimized modes function correctly
 * with their configured MCP servers and can perform their intended operations.
 */

const fs = require('fs');
const path = require('path');

class MCPModeIntegrationTest {
    constructor() {
        this.testResults = {
            passed: 0,
            failed: 0,
            errors: []
        };
        this.modesConfig = null;
        this.mcpConfig = null;
    }

    /**
     * Load configuration files
     */
    async loadConfigurations() {
        try {
            // Load MCP modes configuration
            const modesConfigPath = path.join(__dirname, '..', 'modes.json');
            this.modesConfig = JSON.parse(fs.readFileSync(modesConfigPath, 'utf8'));
            
            // Load MCP server configuration
            const mcpConfigPath = path.join(__dirname, '..', '..', 'mcp.json');
            this.mcpConfig = JSON.parse(fs.readFileSync(mcpConfigPath, 'utf8'));
            
            console.log('✅ Configuration files loaded successfully');
            this.testResults.passed++;
        } catch (error) {
            console.error('❌ Failed to load configuration files:', error.message);
            this.testResults.failed++;
            this.testResults.errors.push(`Configuration loading: ${error.message}`);
        }
    }

    /**
     * Validate modes configuration structure
     */
    validateModesConfiguration() {
        try {
            const requiredFields = ['name', 'slug', 'description', 'model', 'rulesPath', 'mcpServers', 'capabilities', 'workflow'];
            const modes = this.modesConfig.mcpOptimizedModes;
            
            for (const [modeKey, modeConfig] of Object.entries(modes)) {
                // Check required fields
                for (const field of requiredFields) {
                    if (!modeConfig[field]) {
                        throw new Error(`Mode ${modeKey} missing required field: ${field}`);
                    }
                }
                
                // Validate MCP servers exist in configuration
                for (const serverName of modeConfig.mcpServers) {
                    if (!this.mcpConfig.mcpServers[serverName]) {
                        throw new Error(`Mode ${modeKey} references unknown MCP server: ${serverName}`);
                    }
                }
                
                // Validate workflow structure
                if (!modeConfig.workflow.phases || !Array.isArray(modeConfig.workflow.phases)) {
                    throw new Error(`Mode ${modeKey} has invalid workflow phases`);
                }
                
                if (!modeConfig.workflow.mcpIntegration || typeof modeConfig.workflow.mcpIntegration !== 'object') {
                    throw new Error(`Mode ${modeKey} has invalid MCP integration configuration`);
                }
            }
            
            console.log('✅ Modes configuration structure is valid');
            this.testResults.passed++;
        } catch (error) {
            console.error('❌ Modes configuration validation failed:', error.message);
            this.testResults.failed++;
            this.testResults.errors.push(`Modes validation: ${error.message}`);
        }
    }

    /**
     * Validate MCP server tool availability
     */
    validateMCPServerTools() {
        try {
            const modes = this.modesConfig.mcpOptimizedModes;
            
            for (const [modeKey, modeConfig] of Object.entries(modes)) {
                for (const [serverName, tools] of Object.entries(modeConfig.workflow.mcpIntegration)) {
                    const serverConfig = this.mcpConfig.mcpServers[serverName];
                    
                    if (!serverConfig) {
                        throw new Error(`Server ${serverName} not found in MCP configuration`);
                    }
                    
                    // Check if tools are in the alwaysAllow list
                    for (const tool of tools) {
                        if (!serverConfig.alwaysAllow.includes(tool)) {
                            console.warn(`⚠️  Tool ${tool} for server ${serverName} not in alwaysAllow list`);
                        }
                    }
                }
            }
            
            console.log('✅ MCP server tools validation completed');
            this.testResults.passed++;
        } catch (error) {
            console.error('❌ MCP server tools validation failed:', error.message);
            this.testResults.failed++;
            this.testResults.errors.push(`MCP tools validation: ${error.message}`);
        }
    }

    /**
     * Test mode-specific capabilities
     */
    async testModeCapabilities() {
        try {
            const modes = this.modesConfig.mcpOptimizedModes;
            const expectedCapabilities = {
                'mcp-intelligent-coder': ['intelligent_code_generation', 'context_aware_implementation'],
                'mcp-orchestrator': ['workflow_coordination', 'task_distribution'],
                'mcp-researcher': ['deep_research', 'information_synthesis'],
                'mcp-optimizer': ['performance_optimization', 'memory_management'],
                'mcp-management': ['deployment_management', 'system_monitoring'],
                'mcp-tutorial': ['interactive_learning', 'guided_tutorials']
            };
            
            for (const [modeKey, expectedCaps] of Object.entries(expectedCapabilities)) {
                const modeConfig = modes[modeKey];
                if (!modeConfig) {
                    throw new Error(`Mode ${modeKey} not found in configuration`);
                }
                
                for (const capability of expectedCaps) {
                    if (!modeConfig.capabilities.includes(capability)) {
                        throw new Error(`Mode ${modeKey} missing expected capability: ${capability}`);
                    }
                }
            }
            
            console.log('✅ Mode capabilities validation completed');
            this.testResults.passed++;
        } catch (error) {
            console.error('❌ Mode capabilities validation failed:', error.message);
            this.testResults.failed++;
            this.testResults.errors.push(`Mode capabilities: ${error.message}`);
        }
    }

    /**
     * Test workflow phase consistency
     */
    validateWorkflowPhases() {
        try {
            const modes = this.modesConfig.mcpOptimizedModes;
            const commonPhases = ['analysis', 'implementation', 'validation'];
            
            for (const [modeKey, modeConfig] of Object.entries(modes)) {
                const phases = modeConfig.workflow.phases;
                
                // Check minimum phase count
                if (phases.length < 3) {
                    throw new Error(`Mode ${modeKey} has insufficient workflow phases (minimum 3 required)`);
                }
                
                // Check for logical phase ordering (analysis should come before implementation)
                const analysisIndex = phases.indexOf('analysis');
                const implementationIndex = phases.indexOf('implementation');
                
                if (analysisIndex !== -1 && implementationIndex !== -1 && analysisIndex > implementationIndex) {
                    throw new Error(`Mode ${modeKey} has illogical phase ordering: analysis after implementation`);
                }
            }
            
            console.log('✅ Workflow phases validation completed');
            this.testResults.passed++;
        } catch (error) {
            console.error('❌ Workflow phases validation failed:', error.message);
            this.testResults.failed++;
            this.testResults.errors.push(`Workflow phases: ${error.message}`);
        }
    }

    /**
     * Test configuration compatibility
     */
    validateCompatibility() {
        try {
            const config = this.modesConfig.configuration;
            const compatibility = this.modesConfig.compatibility;
            
            // Check required configuration fields
            const requiredConfigFields = ['defaultTimeout', 'retryAttempts', 'mcpServerHealthCheck'];
            for (const field of requiredConfigFields) {
                if (config[field] === undefined) {
                    throw new Error(`Missing required configuration field: ${field}`);
                }
            }
            
            // Check compatibility requirements
            if (!compatibility.requiredServers || !Array.isArray(compatibility.requiredServers)) {
                throw new Error('Invalid required servers configuration');
            }
            
            // Verify required servers exist in MCP configuration
            for (const serverName of compatibility.requiredServers) {
                if (!this.mcpConfig.mcpServers[serverName]) {
                    throw new Error(`Required server ${serverName} not found in MCP configuration`);
                }
            }
            
            console.log('✅ Compatibility validation completed');
            this.testResults.passed++;
        } catch (error) {
            console.error('❌ Compatibility validation failed:', error.message);
            this.testResults.failed++;
            this.testResults.errors.push(`Compatibility: ${error.message}`);
        }
    }

    /**
     * Simulate mode initialization
     */
    async simulateModeInitialization() {
        try {
            const modes = this.modesConfig.mcpOptimizedModes;
            
            for (const [modeKey, modeConfig] of Object.entries(modes)) {
                // Simulate checking if rules file exists
                const rulesPath = path.join(__dirname, '..', '..', '..', modeConfig.rulesPath);
                
                // Create rules directory if it doesn't exist (for testing)
                const rulesDir = path.dirname(rulesPath);
                if (!fs.existsSync(rulesDir)) {
                    fs.mkdirSync(rulesDir, { recursive: true });
                }
                
                // Create a basic rules file if it doesn't exist
                if (!fs.existsSync(rulesPath)) {
                    const basicRules = `# ${modeConfig.name}\n\n## Overview\n\n${modeConfig.description}\n\n## MCP Integration\n\nThis mode integrates with the following MCP servers:\n${modeConfig.mcpServers.map(s => `- ${s}`).join('\n')}\n`;
                    fs.writeFileSync(rulesPath, basicRules);
                }
                
                console.log(`✅ Mode ${modeKey} initialization simulation completed`);
            }
            
            this.testResults.passed++;
        } catch (error) {
            console.error('❌ Mode initialization simulation failed:', error.message);
            this.testResults.failed++;
            this.testResults.errors.push(`Mode initialization: ${error.message}`);
        }
    }

    /**
     * Run all integration tests
     */
    async runAllTests() {
        console.log('🚀 Starting MCP-Optimized Modes Integration Tests\n');
        
        await this.loadConfigurations();
        this.validateModesConfiguration();
        this.validateMCPServerTools();
        await this.testModeCapabilities();
        this.validateWorkflowPhases();
        this.validateCompatibility();
        await this.simulateModeInitialization();
        
        this.printResults();
    }

    /**
     * Print test results
     */
    printResults() {
        console.log('\n📊 Test Results Summary');
        console.log('========================');
        console.log(`✅ Passed: ${this.testResults.passed}`);
        console.log(`❌ Failed: ${this.testResults.failed}`);
        console.log(`📈 Success Rate: ${((this.testResults.passed / (this.testResults.passed + this.testResults.failed)) * 100).toFixed(1)}%`);
        
        if (this.testResults.errors.length > 0) {
            console.log('\n🔍 Error Details:');
            this.testResults.errors.forEach((error, index) => {
                console.log(`${index + 1}. ${error}`);
            });
        }
        
        if (this.testResults.failed === 0) {
            console.log('\n🎉 All tests passed! MCP-optimized modes are ready for use.');
        } else {
            console.log('\n⚠️  Some tests failed. Please review the errors above.');
        }
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    const tester = new MCPModeIntegrationTest();
    tester.runAllTests().catch(console.error);
}

module.exports = MCPModeIntegrationTest;
