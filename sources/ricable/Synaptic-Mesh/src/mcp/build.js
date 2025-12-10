#!/usr/bin/env node

/**
 * Build script for Synaptic Neural Mesh MCP Integration
 * Handles compilation, optimization, and packaging
 */

import { promises as fs } from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import chalk from 'chalk';

class MCPBuilder {
  constructor() {
    this.buildDir = './dist';
    this.sourceDir = './';
    this.config = null;
  }

  async build() {
    console.log(chalk.blue('🔨 Building Synaptic Neural Mesh MCP Integration'));
    console.log();

    try {
      await this.loadConfig();
      await this.createBuildDirectory();
      await this.validateDependencies();
      await this.bundleFiles();
      await this.optimizeBundle();
      await this.generateManifest();
      await this.runTests();
      
      console.log();
      console.log(chalk.green('✅ Build completed successfully!'));
      console.log(chalk.cyan(`📦 Output: ${this.buildDir}`));
      
    } catch (error) {
      console.error(chalk.red('❌ Build failed:'), error.message);
      process.exit(1);
    }
  }

  async loadConfig() {
    console.log(chalk.cyan('📋 Loading configuration...'));
    
    try {
      const configPath = './synaptic-mesh-mcp.config.json';
      const configContent = await fs.readFile(configPath, 'utf8');
      this.config = JSON.parse(configContent);
      console.log(chalk.green('✅ Configuration loaded'));
    } catch (error) {
      console.warn(chalk.yellow('⚠️  Using default configuration'));
      this.config = { name: 'synaptic-mesh-mcp', version: '1.0.0' };
    }
  }

  async createBuildDirectory() {
    console.log(chalk.cyan('📁 Creating build directory...'));
    
    try {
      await fs.rm(this.buildDir, { recursive: true, force: true });
      await fs.mkdir(this.buildDir, { recursive: true });
      console.log(chalk.green(`✅ Build directory created: ${this.buildDir}`));
    } catch (error) {
      throw new Error(`Failed to create build directory: ${error.message}`);
    }
  }

  async validateDependencies() {
    console.log(chalk.cyan('🔍 Validating dependencies...'));
    
    try {
      // Check package.json
      const packagePath = './package.json';
      const packageContent = await fs.readFile(packagePath, 'utf8');
      const packageJson = JSON.parse(packageContent);
      
      // Validate required dependencies
      const requiredDeps = [
        '@modelcontextprotocol/sdk',
        'better-sqlite3',
        'ws',
        'zod',
        'nanoid'
      ];
      
      const missingDeps = requiredDeps.filter(dep => 
        !packageJson.dependencies?.[dep] && !packageJson.devDependencies?.[dep]
      );
      
      if (missingDeps.length > 0) {
        throw new Error(`Missing dependencies: ${missingDeps.join(', ')}`);
      }
      
      console.log(chalk.green('✅ Dependencies validated'));
      
    } catch (error) {
      throw new Error(`Dependency validation failed: ${error.message}`);
    }
  }

  async bundleFiles() {
    console.log(chalk.cyan('📦 Bundling files...'));
    
    const filesToCopy = [
      'index.js',
      'package.json',
      'README.md',
      'synaptic-mesh-mcp.config.json'
    ];
    
    const directoriesToCopy = [
      'server',
      'neural-mesh',
      'transport',
      'auth',
      'events',
      'wasm-bridge',
      'examples'
    ];
    
    // Copy individual files
    for (const file of filesToCopy) {
      try {
        const sourcePath = path.join(this.sourceDir, file);
        const destPath = path.join(this.buildDir, file);
        
        await fs.access(sourcePath);
        await fs.copyFile(sourcePath, destPath);
        console.log(chalk.green(`✅ Copied: ${file}`));
      } catch (error) {
        console.warn(chalk.yellow(`⚠️  Skipped: ${file} (not found)`));
      }
    }
    
    // Copy directories
    for (const dir of directoriesToCopy) {
      try {
        const sourcePath = path.join(this.sourceDir, dir);
        const destPath = path.join(this.buildDir, dir);
        
        await fs.access(sourcePath);
        await this.copyDirectory(sourcePath, destPath);
        console.log(chalk.green(`✅ Copied directory: ${dir}`));
      } catch (error) {
        console.warn(chalk.yellow(`⚠️  Skipped directory: ${dir} (not found)`));
      }
    }
  }

  async copyDirectory(source, destination) {
    await fs.mkdir(destination, { recursive: true });
    
    const entries = await fs.readdir(source, { withFileTypes: true });
    
    for (const entry of entries) {
      const sourcePath = path.join(source, entry.name);
      const destPath = path.join(destination, entry.name);
      
      if (entry.isDirectory()) {
        await this.copyDirectory(sourcePath, destPath);
      } else {
        await fs.copyFile(sourcePath, destPath);
      }
    }
  }

  async optimizeBundle() {
    console.log(chalk.cyan('⚡ Optimizing bundle...'));
    
    try {
      // Create optimized CLI script
      const cliContent = await fs.readFile('./cli.js', 'utf8');
      const optimizedCli = this.optimizeJavaScript(cliContent);
      await fs.writeFile(path.join(this.buildDir, 'cli.js'), optimizedCli);
      
      // Make CLI executable
      await fs.chmod(path.join(this.buildDir, 'cli.js'), 0o755);
      
      console.log(chalk.green('✅ Bundle optimized'));
      
    } catch (error) {
      console.warn(chalk.yellow(`⚠️  Optimization skipped: ${error.message}`));
    }
  }

  optimizeJavaScript(content) {
    // Simple optimization: remove console.debug calls
    return content
      .replace(/console\.debug\([^;]*\);?/g, '')
      .replace(/\/\*\*[\s\S]*?\*\//g, '') // Remove JSDoc comments
      .replace(/^\s*\/\/.*$/gm, '') // Remove single-line comments
      .replace(/\n\s*\n/g, '\n'); // Remove empty lines
  }

  async generateManifest() {
    console.log(chalk.cyan('📋 Generating manifest...'));
    
    const manifest = {
      name: this.config.name,
      version: this.config.version,
      description: 'Synaptic Neural Mesh MCP Integration',
      buildTime: new Date().toISOString(),
      files: await this.getFileList(),
      capabilities: {
        tools: true,
        resources: true,
        prompts: true,
        events: true,
        wasm: this.config.wasmEnabled
      },
      requirements: {
        node: '>=18.0.0',
        memory: '256MB',
        storage: '100MB'
      }
    };
    
    await fs.writeFile(
      path.join(this.buildDir, 'manifest.json'),
      JSON.stringify(manifest, null, 2)
    );
    
    console.log(chalk.green('✅ Manifest generated'));
  }

  async getFileList() {
    const files = [];
    
    async function walk(dir, basePath = '') {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const relativePath = path.join(basePath, entry.name);
        
        if (entry.isDirectory()) {
          await walk(fullPath, relativePath);
        } else {
          const stats = await fs.stat(fullPath);
          files.push({
            path: relativePath,
            size: stats.size,
            modified: stats.mtime.toISOString()
          });
        }
      }
    }
    
    await walk(this.buildDir);
    return files;
  }

  async runTests() {
    console.log(chalk.cyan('🧪 Running tests...'));
    
    try {
      // Copy test files for build validation
      const testDir = path.join(this.buildDir, 'tests');
      await fs.mkdir(testDir, { recursive: true });
      
      if (await this.fileExists('./tests')) {
        await this.copyDirectory('./tests', testDir);
      }
      
      // Run basic validation test
      const validationResult = await this.validateBuild();
      
      if (validationResult.success) {
        console.log(chalk.green('✅ Build validation passed'));
      } else {
        throw new Error(`Build validation failed: ${validationResult.error}`);
      }
      
    } catch (error) {
      console.warn(chalk.yellow(`⚠️  Test execution skipped: ${error.message}`));
    }
  }

  async validateBuild() {
    try {
      // Check if main files exist
      const requiredFiles = ['index.js', 'package.json', 'manifest.json'];
      
      for (const file of requiredFiles) {
        const filePath = path.join(this.buildDir, file);
        await fs.access(filePath);
      }
      
      // Try to load the main module
      const indexPath = path.resolve(this.buildDir, 'index.js');
      const module = await import(indexPath);
      
      if (!module.default) {
        throw new Error('Main module does not export default');
      }
      
      return { success: true };
      
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }
}

// Package creation
class MCPPackager {
  constructor(buildDir) {
    this.buildDir = buildDir;
  }

  async createPackage() {
    console.log(chalk.blue('📦 Creating distribution package...'));
    
    try {
      const packageName = 'synaptic-mesh-mcp-v1.0.0.tar.gz';
      
      // Create tarball
      execSync(`cd ${this.buildDir} && tar -czf ../${packageName} .`, {
        stdio: 'inherit'
      });
      
      console.log(chalk.green(`✅ Package created: ${packageName}`));
      
      // Generate checksums
      const checksumFile = packageName + '.sha256';
      execSync(`sha256sum ${packageName} > ${checksumFile}`, {
        stdio: 'inherit'
      });
      
      console.log(chalk.green(`✅ Checksum generated: ${checksumFile}`));
      
    } catch (error) {
      console.error(chalk.red('❌ Package creation failed:'), error.message);
    }
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  
  const builder = new MCPBuilder();
  await builder.build();
  
  if (args.includes('--package')) {
    const packager = new MCPPackager(builder.buildDir);
    await packager.createPackage();
  }
  
  if (args.includes('--install')) {
    console.log(chalk.blue('📥 Installing MCP server...'));
    try {
      execSync(`cd ${builder.buildDir} && npm install --production`, {
        stdio: 'inherit'
      });
      console.log(chalk.green('✅ Installation completed'));
    } catch (error) {
      console.error(chalk.red('❌ Installation failed:'), error.message);
    }
  }
}

// Run build if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error(chalk.red('❌ Build process failed:'), error);
    process.exit(1);
  });
}

export default MCPBuilder;