#!/usr/bin/env node
/**
 * Post-install script for Synaptic Neural Mesh CLI
 * Sets up necessary permissions and displays welcome message
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Simple color functions for compatibility (no external dependencies)
const colors = {
  cyan: (text) => `\x1b[36m${text}\x1b[0m`,
  green: (text) => `\x1b[32m${text}\x1b[0m`,
  gray: (text) => `\x1b[90m${text}\x1b[0m`,
  red: (text) => `\x1b[31m${text}\x1b[0m`,
  yellow: (text) => `\x1b[33m${text}\x1b[0m`,
  bold: (text) => `\x1b[1m${text}\x1b[0m`
};

console.log('\n' + colors.bold(colors.cyan('🧠 Synaptic Neural Mesh')));
console.log(colors.gray('Setting up CLI...'));

try {
  // Make the bin file executable
  const binPath = path.join(__dirname, '..', 'bin', 'synaptic');
  if (fs.existsSync(binPath)) {
    fs.chmodSync(binPath, '755');
    console.log(colors.green('✓') + ' CLI permissions set');
  }
  
  // Create default directories if they don't exist
  const dataDir = path.join(process.env.HOME || process.env.USERPROFILE, '.synaptic-mesh');
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
    console.log(colors.green('✓') + ' Created configuration directory');
  }
  
  console.log('\n' + colors.green('✨ Installation complete!'));
  console.log('\nGet started with:');
  console.log(colors.cyan('  synaptic init my-project'));
  console.log(colors.cyan('  cd my-project'));
  console.log(colors.cyan('  synaptic start'));
  
  console.log('\n' + colors.gray('Documentation: https://github.com/synaptic-neural-mesh/docs'));
  
} catch (error) {
  console.error(colors.red('Warning:'), 'Post-install setup encountered an issue:', error.message);
  console.log(colors.yellow('The CLI should still work correctly.'));
}