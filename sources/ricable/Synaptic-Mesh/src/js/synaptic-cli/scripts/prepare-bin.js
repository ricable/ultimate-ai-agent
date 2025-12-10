#!/usr/bin/env node

/**
 * Prepare binary files for distribution
 * This script ensures the binary is properly configured for NPX distribution
 */

const fs = require('fs');
const path = require('path');

console.log('🔧 Preparing binary files for distribution...');

const binDir = path.join(__dirname, '..', 'bin');
const libDir = path.join(__dirname, '..', 'lib');

// Ensure bin directory exists
if (!fs.existsSync(binDir)) {
  fs.mkdirSync(binDir, { recursive: true });
}

// Update shebang in binary if needed
const binaryPath = path.join(binDir, 'synaptic-mesh');
if (fs.existsSync(binaryPath)) {
  let content = fs.readFileSync(binaryPath, 'utf-8');
  
  // Ensure proper shebang
  if (!content.startsWith('#!/usr/bin/env node')) {
    content = '#!/usr/bin/env node\n' + content;
    fs.writeFileSync(binaryPath, content);
  }
  
  // Make executable
  fs.chmodSync(binaryPath, 0o755);
  console.log('✅ Binary prepared and made executable');
} else {
  console.warn('⚠️  Binary file not found at', binaryPath);
}

// Create symlink for 'synaptic' alias
const synapticPath = path.join(binDir, 'synaptic');
if (!fs.existsSync(synapticPath)) {
  try {
    fs.symlinkSync('./synaptic-mesh', synapticPath);
    console.log('✅ Created symlink for synaptic alias');
  } catch (err) {
    // On Windows, copy instead of symlink
    if (fs.existsSync(binaryPath)) {
      fs.copyFileSync(binaryPath, synapticPath);
      fs.chmodSync(synapticPath, 0o755);
      console.log('✅ Created copy for synaptic alias (Windows)');
    }
  }
}

console.log('🎉 Binary preparation complete!');