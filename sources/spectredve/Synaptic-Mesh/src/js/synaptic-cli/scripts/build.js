#!/usr/bin/env node

/**
 * Build script for Synaptic Neural Mesh CLI
 */

const fs = require('fs-extra');
const path = require('path');

async function build() {
  console.log('Building Synaptic Neural Mesh CLI...');
  
  try {
    // Ensure dist directory exists
    await fs.ensureDir('dist');
    
    // Copy source files
    await fs.copy('src', 'dist', {
      filter: (src) => {
        // Skip TypeScript files, tests, and temp files
        return !src.endsWith('.ts') && !src.includes('.test.') && !src.includes('.tmp');
      }
    });
    
    // Make bin files executable
    const binDir = path.join('dist', 'bin');
    if (await fs.pathExists(binDir)) {
      const binFiles = await fs.readdir(binDir);
      for (const file of binFiles) {
        const filePath = path.join(binDir, file);
        await fs.chmod(filePath, '755');
      }
    }
    
    console.log('Build completed successfully!');
    
  } catch (error) {
    console.error('Build failed:', error.message);
    process.exit(1);
  }
}

build();