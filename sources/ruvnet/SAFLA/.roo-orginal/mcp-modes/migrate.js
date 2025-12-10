#!/usr/bin/env node

/**
 * MCP-Optimized Modes Migration Script
 * 
 * This script helps users transition from standard SAFLA modes to MCP-optimized modes.
 * It provides backup, configuration updates, and validation to ensure a smooth migration.
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

class MCPModesMigration {
    constructor() {
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        this.backupDir = '.roo/backup';
        this.timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        this.migrationLog = [];
    }

    /**
     * Ask user a question and return the response
     */
    async askQuestion(question) {
        return new Promise((resolve) => {
            this.rl.question(question, (answer) => {
                resolve(answer.trim());
            });
        });
    }

    /**
     * Log migration steps
     */
    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const logEntry = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
        this.migrationLog.push(logEntry);
        
        const emoji = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        
        console.log(`${emoji[type] || 'ℹ️'} ${message}`);
    }

    /**
     * Create backup directory and backup existing configurations
     */
    async createBackup() {
        try {
            const backupPath = path.join(this.backupDir, `migration-${this.timestamp}`);
            
            if (!fs.existsSync(this.backupDir)) {
                fs.mkdirSync(this.backupDir, { recursive: true });
            }
            
            fs.mkdirSync(backupPath, { recursive: true });
            
            // Backup existing MCP configuration
            const mcpConfigPath = '.roo/mcp.json';
            if (fs.existsSync(mcpConfigPath)) {
                fs.copyFileSync(mcpConfigPath, path.join(backupPath, 'mcp.json'));
                this.log(`Backed up MCP configuration to ${backupPath}/mcp.json`, 'success');
            }
            
            // Backup existing rules directories
            const rulesDir = '.roo/rules';
            if (fs.existsSync(rulesDir)) {
                this.copyDirectory(rulesDir, path.join(backupPath, 'rules'));
                this.log(`Backed up rules directory to ${backupPath}/rules`, 'success');
            }
            
            // Create migration log file
            const logPath = path.join(backupPath, 'migration.log');
            fs.writeFileSync(logPath, `Migration started at ${new Date().toISOString()}\n`);
            
            this.log(`Backup created successfully at ${backupPath}`, 'success');
            return backupPath;
            
        } catch (error) {
            this.log(`Failed to create backup: ${error.message}`, 'error');
            throw error;
        }
    }

    /**
     * Copy directory recursively
     */
    copyDirectory(src, dest) {
        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest, { recursive: true });
        }
        
        const items = fs.readdirSync(src);
        
        for (const item of items) {
            const srcPath = path.join(src, item);
            const destPath = path.join(dest, item);
            
            if (fs.statSync(srcPath).isDirectory()) {
                this.copyDirectory(srcPath, destPath);
            } else {
                fs.copyFileSync(srcPath, destPath);
            }
        }
    }

    /**
     * Check current system configuration
     */
    async checkCurrentConfiguration() {
        this.log('Checking current system configuration...', 'info');
        
        const checks = {
            mcpConfig: fs.existsSync('.roo/mcp.json'),
            rulesDir: fs.existsSync('.roo/rules'),
            mcpModesDir: fs.existsSync('.roo/mcp-modes'),
            nodeJs: this.checkNodeJs()
        };
        
        this.log(`MCP Configuration: ${checks.mcpConfig ? 'Found' : 'Missing'}`, checks.mcpConfig ? 'success' : 'warning');
        this.log(`Rules Directory: ${checks.rulesDir ? 'Found' : 'Missing'}`, checks.rulesDir ? 'success' : 'warning');
        this.log(`MCP Modes Directory: ${checks.mcpModesDir ? 'Found' : 'Missing'}`, checks.mcpModesDir ? 'success' : 'warning');
        this.log(`Node.js: ${checks.nodeJs ? 'Available' : 'Missing'}`, checks.nodeJs ? 'success' : 'error');
        
        return checks;
    }

    /**
     * Check if Node.js is available
     */
    checkNodeJs() {
        try {
            require('child_process').execSync('node --version', { stdio: 'ignore' });
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Validate MCP server configurations
     */
    async validateMCPServers() {
        try {
            const mcpConfigPath = '.roo/mcp.json';
            if (!fs.existsSync(mcpConfigPath)) {
                this.log('MCP configuration file not found', 'warning');
                return false;
            }
            
            const mcpConfig = JSON.parse(fs.readFileSync(mcpConfigPath, 'utf8'));
            const requiredServers = ['safla'];
            const optionalServers = ['context7', 'perplexity'];
            
            let allRequired = true;
            
            for (const server of requiredServers) {
                if (mcpConfig.mcpServers && mcpConfig.mcpServers[server]) {
                    this.log(`Required server '${server}' is configured`, 'success');
                } else {
                    this.log(`Required server '${server}' is missing`, 'error');
                    allRequired = false;
                }
            }
            
            for (const server of optionalServers) {
                if (mcpConfig.mcpServers && mcpConfig.mcpServers[server]) {
                    this.log(`Optional server '${server}' is configured`, 'success');
                } else {
                    this.log(`Optional server '${server}' is not configured`, 'warning');
                }
            }
            
            return allRequired;
            
        } catch (error) {
            this.log(`Failed to validate MCP servers: ${error.message}`, 'error');
            return false;
        }
    }

    /**
     * Create mode mapping from standard to MCP-optimized modes
     */
    getModeMapping() {
        return {
            'code': 'mcp-intelligent-coder',
            'orchestrator': 'mcp-orchestrator',
            'deep-research': 'mcp-researcher',
            'architect': 'mcp-optimizer',
            'mcp': 'mcp-management',
            'tutorial': 'mcp-tutorial'
        };
    }

    /**
     * Update user configurations and scripts
     */
    async updateUserConfigurations() {
        this.log('Updating user configurations...', 'info');
        
        const modeMapping = this.getModeMapping();
        let updatedFiles = 0;
        
        // Search for files that might contain mode references
        const searchPatterns = [
            { dir: 'scripts', extensions: ['.sh', '.js', '.py'] },
            { dir: 'docs', extensions: ['.md', '.txt'] },
            { dir: '.', extensions: ['.md', '.json'], depth: 1 }
        ];
        
        for (const pattern of searchPatterns) {
            if (fs.existsSync(pattern.dir)) {
                const files = this.findFilesWithExtensions(pattern.dir, pattern.extensions, pattern.depth);
                
                for (const file of files) {
                    if (this.updateModeReferences(file, modeMapping)) {
                        updatedFiles++;
                    }
                }
            }
        }
        
        this.log(`Updated mode references in ${updatedFiles} files`, 'success');
        return updatedFiles;
    }

    /**
     * Find files with specific extensions
     */
    findFilesWithExtensions(dir, extensions, maxDepth = Infinity, currentDepth = 0) {
        const files = [];
        
        if (currentDepth >= maxDepth) {
            return files;
        }
        
        try {
            const items = fs.readdirSync(dir);
            
            for (const item of items) {
                const fullPath = path.join(dir, item);
                const stat = fs.statSync(fullPath);
                
                if (stat.isDirectory() && currentDepth < maxDepth - 1) {
                    files.push(...this.findFilesWithExtensions(fullPath, extensions, maxDepth, currentDepth + 1));
                } else if (stat.isFile()) {
                    const ext = path.extname(item);
                    if (extensions.includes(ext)) {
                        files.push(fullPath);
                    }
                }
            }
        } catch (error) {
            // Ignore permission errors
        }
        
        return files;
    }

    /**
     * Update mode references in a file
     */
    updateModeReferences(filePath, modeMapping) {
        try {
            let content = fs.readFileSync(filePath, 'utf8');
            let updated = false;
            
            for (const [oldMode, newMode] of Object.entries(modeMapping)) {
                const patterns = [
                    new RegExp(`new_task:\\s*${oldMode}`, 'g'),
                    new RegExp(`"mode":\\s*"${oldMode}"`, 'g'),
                    new RegExp(`'mode':\\s*'${oldMode}'`, 'g'),
                    new RegExp(`mode=${oldMode}`, 'g')
                ];
                
                for (const pattern of patterns) {
                    if (pattern.test(content)) {
                        content = content.replace(pattern, (match) => {
                            return match.replace(oldMode, newMode);
                        });
                        updated = true;
                    }
                }
            }
            
            if (updated) {
                fs.writeFileSync(filePath, content);
                this.log(`Updated mode references in ${filePath}`, 'success');
                return true;
            }
            
        } catch (error) {
            this.log(`Failed to update ${filePath}: ${error.message}`, 'warning');
        }
        
        return false;
    }

    /**
     * Run integration tests
     */
    async runIntegrationTests() {
        this.log('Running integration tests...', 'info');
        
        try {
            const testPath = '.roo/mcp-modes/tests/integration_test.js';
            if (!fs.existsSync(testPath)) {
                this.log('Integration test file not found', 'warning');
                return false;
            }
            
            const { spawn } = require('child_process');
            
            return new Promise((resolve) => {
                const testProcess = spawn('node', [testPath], {
                    stdio: 'pipe',
                    cwd: process.cwd()
                });
                
                let output = '';
                let errorOutput = '';
                
                testProcess.stdout.on('data', (data) => {
                    output += data.toString();
                });
                
                testProcess.stderr.on('data', (data) => {
                    errorOutput += data.toString();
                });
                
                testProcess.on('close', (code) => {
                    if (code === 0) {
                        this.log('Integration tests passed successfully', 'success');
                        console.log(output);
                        resolve(true);
                    } else {
                        this.log('Integration tests failed', 'error');
                        console.log(output);
                        console.error(errorOutput);
                        resolve(false);
                    }
                });
            });
            
        } catch (error) {
            this.log(`Failed to run integration tests: ${error.message}`, 'error');
            return false;
        }
    }

    /**
     * Generate migration report
     */
    generateMigrationReport(backupPath) {
        const reportPath = path.join(backupPath, 'migration-report.md');
        const modeMapping = this.getModeMapping();
        
        const report = `# MCP-Optimized Modes Migration Report

## Migration Summary

**Date**: ${new Date().toISOString()}
**Backup Location**: ${backupPath}

## Mode Mapping

The following standard modes have been mapped to MCP-optimized modes:

${Object.entries(modeMapping).map(([old, new_]) => `- \`${old}\` → \`${new_}\``).join('\n')}

## Migration Log

${this.migrationLog.join('\n')}

## Next Steps

1. **Test the new modes**: Try using the MCP-optimized modes with \`new_task: <mode-name>\`
2. **Update documentation**: Review and update any project documentation that references the old mode names
3. **Train team members**: Ensure all team members are aware of the new mode names and capabilities
4. **Monitor performance**: Keep an eye on the enhanced capabilities provided by MCP integration

## Rollback Instructions

If you need to rollback the migration:

1. Stop any running MCP-optimized mode sessions
2. Restore the backup files from: \`${backupPath}\`
3. Restart the SAFLA system
4. Use the standard mode names

## Support

For issues with the migration or MCP-optimized modes:

- Check the integration test results
- Review the MCP server configurations
- Consult the MCP-optimized modes documentation
- Report issues through standard SAFLA channels
`;

        fs.writeFileSync(reportPath, report);
        this.log(`Migration report generated: ${reportPath}`, 'success');
        
        return reportPath;
    }

    /**
     * Main migration workflow
     */
    async runMigration() {
        console.log('🚀 MCP-Optimized Modes Migration Tool\n');
        
        try {
            // Check current configuration
            const configCheck = await this.checkCurrentConfiguration();
            
            if (!configCheck.nodeJs) {
                this.log('Node.js is required for this migration. Please install Node.js and try again.', 'error');
                return;
            }
            
            // Ask for user confirmation
            const proceed = await this.askQuestion('\nDo you want to proceed with the migration? (y/N): ');
            if (proceed.toLowerCase() !== 'y' && proceed.toLowerCase() !== 'yes') {
                this.log('Migration cancelled by user', 'info');
                return;
            }
            
            // Create backup
            this.log('\n📦 Creating backup...', 'info');
            const backupPath = await this.createBackup();
            
            // Validate MCP servers
            this.log('\n🔍 Validating MCP server configuration...', 'info');
            const mcpValid = await this.validateMCPServers();
            
            if (!mcpValid) {
                const continueAnyway = await this.askQuestion('\nMCP server validation failed. Continue anyway? (y/N): ');
                if (continueAnyway.toLowerCase() !== 'y' && continueAnyway.toLowerCase() !== 'yes') {
                    this.log('Migration cancelled due to MCP server validation failure', 'warning');
                    return;
                }
            }
            
            // Update user configurations
            this.log('\n📝 Updating configuration files...', 'info');
            await this.updateUserConfigurations();
            
            // Run integration tests
            this.log('\n🧪 Running integration tests...', 'info');
            const testsPass = await this.runIntegrationTests();
            
            // Generate migration report
            this.log('\n📊 Generating migration report...', 'info');
            const reportPath = this.generateMigrationReport(backupPath);
            
            // Final summary
            console.log('\n🎉 Migration completed successfully!');
            console.log('\n📋 Summary:');
            console.log(`   • Backup created: ${backupPath}`);
            console.log(`   • Migration report: ${reportPath}`);
            console.log(`   • Integration tests: ${testsPass ? 'PASSED' : 'FAILED'}`);
            
            console.log('\n🚀 You can now use MCP-optimized modes:');
            const modeMapping = this.getModeMapping();
            Object.entries(modeMapping).forEach(([old, new_]) => {
                console.log(`   • new_task: ${new_} (was: ${old})`);
            });
            
            if (!testsPass) {
                console.log('\n⚠️  Note: Integration tests failed. Please review the test output and fix any issues before using the MCP-optimized modes.');
            }
            
        } catch (error) {
            this.log(`Migration failed: ${error.message}`, 'error');
            console.log('\n❌ Migration failed. Please check the error messages above and try again.');
        } finally {
            this.rl.close();
        }
    }
}

// Run migration if this file is executed directly
if (require.main === module) {
    const migration = new MCPModesMigration();
    migration.runMigration().catch(console.error);
}

module.exports = MCPModesMigration;
