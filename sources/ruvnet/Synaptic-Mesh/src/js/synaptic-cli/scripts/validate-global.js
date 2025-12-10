#!/usr/bin/env node

/**
 * Global Package Distribution Validator
 * Tests NPX installation and cross-platform compatibility
 */

const { spawn } = require('child_process');
const fs = require('fs-extra');
const path = require('path');
const os = require('os');

class GlobalValidator {
    constructor() {
        this.platform = os.platform();
        this.arch = os.arch();
        this.tempDir = path.join(os.tmpdir(), 'synaptic-mesh-validation');
        this.packageTarball = null;
        this.results = {
            platform: this.platform,
            arch: this.arch,
            node: process.version,
            npm: null,
            tests: {},
            errors: [],
            warnings: []
        };
    }

    async run() {
        console.log('🧠 Synaptic Neural Mesh - Global Distribution Validator');
        console.log(`Platform: ${this.platform} (${this.arch})`);
        console.log(`Node.js: ${process.version}`);
        console.log('=====================================\n');

        try {
            await this.setup();
            await this.validateNpmVersion();
            await this.findPackageTarball();
            await this.testLocalInstall();
            await this.testNpxExecution();
            await this.testCrossCompatibility();
            await this.cleanup();
            
            this.generateReport();
        } catch (error) {
            this.results.errors.push(error.message);
            console.error('❌ Validation failed:', error.message);
            process.exit(1);
        }
    }

    async setup() {
        console.log('🔧 Setting up validation environment...');
        await fs.ensureDir(this.tempDir);
        process.chdir(this.tempDir);
    }

    async validateNpmVersion() {
        console.log('📦 Checking npm version...');
        const npmVersion = await this.executeCommand('npm', ['--version']);
        this.results.npm = npmVersion.trim();
        console.log(`✅ npm version: ${this.results.npm}`);
    }

    async findPackageTarball() {
        console.log('🔍 Finding package tarball...');
        const projectRoot = path.dirname(path.dirname(__dirname));
        const files = await fs.readdir(projectRoot);
        
        this.packageTarball = files.find(file => 
            file.startsWith('synaptic-mesh-') && file.endsWith('.tgz')
        );
        
        if (!this.packageTarball) {
            throw new Error('Package tarball not found. Run "npm pack" first.');
        }
        
        this.packageTarball = path.join(projectRoot, this.packageTarball);
        console.log(`✅ Found tarball: ${this.packageTarball}`);
    }

    async testLocalInstall() {
        console.log('📥 Testing local installation...');
        
        try {
            await this.executeCommand('npm', ['install', this.packageTarball]);
            this.results.tests.localInstall = 'passed';
            console.log('✅ Local installation successful');
        } catch (error) {
            this.results.tests.localInstall = 'failed';
            throw new Error(`Local installation failed: ${error.message}`);
        }
    }

    async testNpxExecution() {
        console.log('🚀 Testing npx execution...');
        
        const tests = [
            { name: 'npx synaptic-mesh --version', args: ['synaptic-mesh', '--version'] },
            { name: 'npx synaptic-mesh --help', args: ['synaptic-mesh', '--help'] },
            { name: 'npx synaptic-mesh init --dry-run', args: ['synaptic-mesh', 'init', '--dry-run'] }
        ];

        for (const test of tests) {
            try {
                console.log(`  Testing: ${test.name}`);
                const output = await this.executeCommand('npx', test.args, { timeout: 30000 });
                this.results.tests[`npx_${test.name.replace(/[^a-zA-Z0-9]/g, '_')}`] = 'passed';
                console.log(`  ✅ ${test.name} - passed`);
            } catch (error) {
                this.results.tests[`npx_${test.name.replace(/[^a-zA-Z0-9]/g, '_')}`] = 'failed';
                this.results.warnings.push(`${test.name} failed: ${error.message}`);
                console.log(`  ⚠️ ${test.name} - failed: ${error.message}`);
            }
        }
    }

    async testCrossCompatibility() {
        console.log('🔀 Testing cross-platform compatibility...');
        
        const compatibility = {
            wasmSupport: await this.checkWasmSupport(),
            binaryExecution: await this.checkBinaryExecution(),
            pathResolution: await this.checkPathResolution(),
            permissions: await this.checkPermissions()
        };

        this.results.tests.compatibility = compatibility;
        
        Object.entries(compatibility).forEach(([key, value]) => {
            console.log(`  ${value ? '✅' : '❌'} ${key}: ${value ? 'supported' : 'not supported'}`);
        });
    }

    async checkWasmSupport() {
        try {
            const wasmTest = `
                const wasmCode = new Uint8Array([
                    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00
                ]);
                WebAssembly.validate(wasmCode);
                console.log('wasm-supported');
            `;
            
            await this.executeCommand('node', ['-e', wasmTest]);
            return true;
        } catch {
            return false;
        }
    }

    async checkBinaryExecution() {
        try {
            const binPath = path.join('node_modules', 'synaptic-mesh', 'bin', 'synaptic-mesh');
            if (await fs.pathExists(binPath)) {
                await this.executeCommand('node', [binPath, '--version'], { timeout: 10000 });
                return true;
            }
            return false;
        } catch {
            return false;
        }
    }

    async checkPathResolution() {
        try {
            const which = this.platform === 'win32' ? 'where' : 'which';
            await this.executeCommand(which, ['node']);
            return true;
        } catch {
            return false;
        }
    }

    async checkPermissions() {
        try {
            const testFile = path.join(this.tempDir, 'perm-test.txt');
            await fs.writeFile(testFile, 'test');
            await fs.unlink(testFile);
            return true;
        } catch {
            return false;
        }
    }

    async cleanup() {
        console.log('🧹 Cleaning up...');
        try {
            await fs.remove(this.tempDir);
        } catch (error) {
            this.results.warnings.push(`Cleanup failed: ${error.message}`);
        }
    }

    generateReport() {
        console.log('\n📊 Validation Report');
        console.log('====================');
        
        const passedTests = Object.values(this.results.tests).filter(v => v === 'passed' || v === true).length;
        const totalTests = Object.keys(this.results.tests).length;
        
        console.log(`✅ Tests passed: ${passedTests}/${totalTests}`);
        console.log(`⚠️ Warnings: ${this.results.warnings.length}`);
        console.log(`❌ Errors: ${this.results.errors.length}`);
        
        if (this.results.warnings.length > 0) {
            console.log('\nWarnings:');
            this.results.warnings.forEach(warning => console.log(`  ⚠️ ${warning}`));
        }
        
        if (this.results.errors.length > 0) {
            console.log('\nErrors:');
            this.results.errors.forEach(error => console.log(`  ❌ ${error}`));
        }
        
        // Save detailed report
        const reportPath = path.join(process.cwd(), 'global-validation-report.json');
        fs.writeFileSync(reportPath, JSON.stringify(this.results, null, 2));
        console.log(`\n📝 Detailed report saved to: ${reportPath}`);
        
        const success = this.results.errors.length === 0 && passedTests > totalTests * 0.8;
        console.log(`\n${success ? '🎉' : '💥'} Validation ${success ? 'PASSED' : 'FAILED'}`);
        
        if (!success) {
            process.exit(1);
        }
    }

    executeCommand(command, args = [], options = {}) {
        return new Promise((resolve, reject) => {
            const timeout = options.timeout || 15000;
            let output = '';
            let errorOutput = '';
            
            const child = spawn(command, args, {
                stdio: 'pipe',
                shell: this.platform === 'win32'
            });
            
            const timer = setTimeout(() => {
                child.kill();
                reject(new Error(`Command timeout: ${command} ${args.join(' ')}`));
            }, timeout);
            
            child.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            child.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });
            
            child.on('close', (code) => {
                clearTimeout(timer);
                if (code === 0) {
                    resolve(output);
                } else {
                    reject(new Error(`Command failed (${code}): ${errorOutput || output}`));
                }
            });
            
            child.on('error', (error) => {
                clearTimeout(timer);
                reject(error);
            });
        });
    }
}

// Run validation if called directly
if (require.main === module) {
    new GlobalValidator().run().catch(console.error);
}

module.exports = GlobalValidator;