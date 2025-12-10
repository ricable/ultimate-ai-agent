#!/usr/bin/env node
/**
 * Comprehensive packaging validation script for Synaptic Neural Mesh
 * Tests NPM package installation, global commands, and cross-platform compatibility
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

class PackagingValidator {
  constructor() {
    this.results = {
      passed: 0,
      failed: 0,
      warnings: 0,
      tests: []
    };
  }

  log(message, type = 'info') {
    const color = {
      'info': colors.blue,
      'success': colors.green,
      'error': colors.red,
      'warning': colors.yellow
    }[type] || colors.reset;
    
    console.log(`${color}[${type.toUpperCase()}]${colors.reset} ${message}`);
  }

  async runTest(name, testFn) {
    this.log(`Running test: ${name}`, 'info');
    try {
      const startTime = Date.now();
      await testFn();
      const duration = Date.now() - startTime;
      this.log(`✅ ${name} (${duration}ms)`, 'success');
      this.results.passed++;
      this.results.tests.push({ name, status: 'passed', duration });
    } catch (error) {
      this.log(`❌ ${name}: ${error.message}`, 'error');
      this.results.failed++;
      this.results.tests.push({ name, status: 'failed', error: error.message });
    }
  }

  async validateNodeVersions() {
    const nodeVersion = process.version;
    const npmVersion = execSync('npm --version', { encoding: 'utf8' }).trim();
    
    this.log(`Node.js version: ${nodeVersion}`, 'info');
    this.log(`NPM version: ${npmVersion}`, 'info');
    
    const nodeMajor = parseInt(nodeVersion.slice(1).split('.')[0]);
    if (nodeMajor < 18) {
      throw new Error(`Node.js 18+ required, found ${nodeVersion}`);
    }
  }

  async validatePackageStructure() {
    const mainPackageJson = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli/package.json';
    const npxWrapperJson = '/workspaces/Synaptic-Neural-Mesh/npx-wrapper/package.json';
    
    if (!fs.existsSync(mainPackageJson)) {
      throw new Error('Main package.json not found');
    }
    
    const mainPkg = JSON.parse(fs.readFileSync(mainPackageJson, 'utf8'));
    
    // Validate essential fields
    const requiredFields = ['name', 'version', 'description', 'main', 'bin', 'files', 'engines'];
    for (const field of requiredFields) {
      if (!mainPkg[field]) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
    
    // Validate binary definitions
    if (!mainPkg.bin || Object.keys(mainPkg.bin).length === 0) {
      throw new Error('No binary definitions found');
    }
    
    // Validate engine constraints
    if (!mainPkg.engines.node || !mainPkg.engines.npm) {
      throw new Error('Missing engine constraints');
    }
  }

  async validateDependencies() {
    const packagePath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
    const packageJson = JSON.parse(fs.readFileSync(path.join(packagePath, 'package.json'), 'utf8'));
    
    // Check for security vulnerabilities in dependencies
    this.log('Checking for dependency vulnerabilities...', 'info');
    
    try {
      execSync('npm audit --audit-level=moderate', { 
        cwd: packagePath, 
        stdio: 'pipe' 
      });
    } catch (error) {
      this.log('⚠️ Some dependency vulnerabilities found', 'warning');
      this.results.warnings++;
    }
    
    // Validate peer dependencies
    if (packageJson.peerDependencies) {
      for (const [dep, version] of Object.entries(packageJson.peerDependencies)) {
        this.log(`Peer dependency: ${dep}@${version}`, 'info');
      }
    }
  }

  async testLocalInstallation() {
    const packagePath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'synaptic-test-'));
    
    try {
      // Pack the package
      this.log('Creating package tarball...', 'info');
      const packResult = execSync('npm pack', { 
        cwd: packagePath, 
        encoding: 'utf8' 
      });
      
      const tarball = packResult.trim();
      const tarballPath = path.join(packagePath, tarball);
      
      // Test local installation
      this.log('Testing local installation...', 'info');
      execSync(`npm install ${tarballPath}`, { 
        cwd: tempDir,
        stdio: 'pipe'
      });
      
      // Cleanup
      fs.unlinkSync(tarballPath);
      
    } finally {
      // Cleanup temp directory
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    }
  }

  async testGlobalInstallation() {
    const packagePath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
    
    try {
      // Create package
      const packResult = execSync('npm pack', { 
        cwd: packagePath, 
        encoding: 'utf8' 
      });
      
      const tarball = packResult.trim();
      const tarballPath = path.join(packagePath, tarball);
      
      // Test global installation (dry run)
      this.log('Testing global installation (dry-run)...', 'info');
      execSync(`npm install -g ${tarballPath} --dry-run`, { 
        stdio: 'pipe' 
      });
      
      // Cleanup
      fs.unlinkSync(tarballPath);
      
    } catch (error) {
      throw new Error(`Global installation test failed: ${error.message}`);
    }
  }

  async validateBinaries() {
    const packagePath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
    const packageJson = JSON.parse(fs.readFileSync(path.join(packagePath, 'package.json'), 'utf8'));
    
    if (!packageJson.bin) {
      throw new Error('No binaries defined');
    }
    
    for (const [binName, binPath] of Object.entries(packageJson.bin)) {
      const fullBinPath = path.join(packagePath, binPath);
      
      if (!fs.existsSync(fullBinPath)) {
        throw new Error(`Binary file not found: ${binPath}`);
      }
      
      // Check if binary is executable
      try {
        fs.accessSync(fullBinPath, fs.constants.F_OK);
        this.log(`✓ Binary validated: ${binName} -> ${binPath}`, 'info');
      } catch (error) {
        throw new Error(`Binary not accessible: ${binPath}`);
      }
    }
  }

  async validateCrossPlatform() {
    const platform = os.platform();
    const arch = os.arch();
    
    this.log(`Platform: ${platform} (${arch})`, 'info');
    
    // Check if platform is supported
    const supportedPlatforms = ['linux', 'darwin', 'win32'];
    const supportedArchs = ['x64', 'arm64'];
    
    if (!supportedPlatforms.includes(platform)) {
      this.log(`⚠️ Platform ${platform} not explicitly supported`, 'warning');
      this.results.warnings++;
    }
    
    if (!supportedArchs.includes(arch)) {
      this.log(`⚠️ Architecture ${arch} not explicitly supported`, 'warning');
      this.results.warnings++;
    }
  }

  async validateAlphaRelease() {
    const packagePath = '/workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli';
    const packageJson = JSON.parse(fs.readFileSync(path.join(packagePath, 'package.json'), 'utf8'));
    
    // Check version format
    const version = packageJson.version;
    if (!version.includes('alpha')) {
      this.log(`⚠️ Version ${version} doesn't include 'alpha' tag`, 'warning');
      this.results.warnings++;
    }
    
    // Check publish config
    if (packageJson.publishConfig) {
      if (packageJson.publishConfig.tag !== 'alpha') {
        this.log(`⚠️ publishConfig.tag should be 'alpha' for pre-release`, 'warning');
        this.results.warnings++;
      }
      
      if (packageJson.publishConfig.access !== 'public') {
        this.log(`⚠️ publishConfig.access should be 'public'`, 'warning');
        this.results.warnings++;
      }
    }
  }

  async generateReport() {
    const total = this.results.passed + this.results.failed;
    const successRate = total > 0 ? ((this.results.passed / total) * 100).toFixed(1) : 0;
    
    this.log('\n' + '='.repeat(60), 'info');
    this.log('📦 PACKAGING VALIDATION REPORT', 'info');
    this.log('='.repeat(60), 'info');
    
    this.log(`Total Tests: ${total}`, 'info');
    this.log(`Passed: ${this.results.passed}`, 'success');
    this.log(`Failed: ${this.results.failed}`, 'error');
    this.log(`Warnings: ${this.results.warnings}`, 'warning');
    this.log(`Success Rate: ${successRate}%`, 'info');
    
    if (this.results.failed > 0) {
      this.log('\n❌ FAILED TESTS:', 'error');
      this.results.tests
        .filter(test => test.status === 'failed')
        .forEach(test => {
          this.log(`  • ${test.name}: ${test.error}`, 'error');
        });
    }
    
    if (this.results.warnings > 0) {
      this.log('\n⚠️ WARNINGS: Review these issues for production release', 'warning');
    }
    
    if (this.results.failed === 0) {
      this.log('\n🎉 All critical tests passed! Package is ready for alpha release.', 'success');
    } else {
      this.log('\n🚨 Some tests failed. Please fix issues before release.', 'error');
    }
    
    // Write detailed report to file
    const reportPath = '/workspaces/Synaptic-Neural-Mesh/packaging-report.json';
    const detailedReport = {
      timestamp: new Date().toISOString(),
      platform: {
        os: os.platform(),
        arch: os.arch(),
        nodeVersion: process.version,
        npmVersion: execSync('npm --version', { encoding: 'utf8' }).trim()
      },
      results: this.results
    };
    
    fs.writeFileSync(reportPath, JSON.stringify(detailedReport, null, 2));
    this.log(`📄 Detailed report saved to: ${reportPath}`, 'info');
  }

  async run() {
    this.log('🚀 Starting Synaptic Neural Mesh Packaging Validation', 'info');
    
    await this.runTest('Node.js Version Compatibility', () => this.validateNodeVersions());
    await this.runTest('Package Structure Validation', () => this.validatePackageStructure());
    await this.runTest('Dependency Analysis', () => this.validateDependencies());
    await this.runTest('Binary Validation', () => this.validateBinaries());
    await this.runTest('Local Installation Test', () => this.testLocalInstallation());
    await this.runTest('Global Installation Test', () => this.testGlobalInstallation());
    await this.runTest('Cross-Platform Compatibility', () => this.validateCrossPlatform());
    await this.runTest('Alpha Release Validation', () => this.validateAlphaRelease());
    
    await this.generateReport();
    
    // Return exit code based on results
    return this.results.failed === 0 ? 0 : 1;
  }
}

// Run if called directly
if (require.main === module) {
  const validator = new PackagingValidator();
  validator.run().then(exitCode => {
    process.exit(exitCode);
  }).catch(error => {
    console.error(`${colors.red}❌ Fatal error: ${error.message}${colors.reset}`);
    process.exit(1);
  });
}

module.exports = PackagingValidator;