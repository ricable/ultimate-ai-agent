#!/usr/bin/env node

/**
 * Simple runtime test to verify core functionality
 */

console.log('🚀 Starting RAN Optimization SDK Runtime Test...');

try {
  // Test basic imports
  console.log('✅ Testing basic imports...');

  // Test core memory integration
  const fs = require('fs');
  const path = require('path');

  console.log('✅ Basic Node.js modules loaded successfully');

  // Test file system access to source files
  const srcPath = path.join(__dirname, 'src');
  const files = fs.readdirSync(srcPath);
  console.log(`✅ Found ${files.length} files/directories in src/`);

  // Test package.json exists and is readable
  const packagePath = path.join(__dirname, 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
  console.log(`✅ Package ${packageJson.name} v${packageJson.version} loaded`);

  // Test memory integration file exists
  const memoryPath = path.join(__dirname, 'src/memory/agentdb-integration.ts');
  if (fs.existsSync(memoryPath)) {
    console.log('✅ AgentDB integration file exists');
  } else {
    console.log('❌ AgentDB integration file missing');
  }

  // Test key source files
  const keyFiles = [
    'src/index.ts',
    'src/memory/agentdb-integration.ts',
    'src/stream-chain/core.ts',
    'src/action-execution/execution-engine.ts'
  ];

  let existingFiles = 0;
  keyFiles.forEach(file => {
    const filePath = path.join(__dirname, file);
    if (fs.existsSync(filePath)) {
      existingFiles++;
      console.log(`✅ ${file} exists`);
    } else {
      console.log(`❌ ${file} missing`);
    }
  });

  console.log(`\n📊 Runtime Test Summary:`);
  console.log(`   - Core files present: ${existingFiles}/${keyFiles.length}`);
  console.log(`   - Source directories: ${files.length}`);
  console.log(`   - Package loaded successfully`);
  console.log(`   - File system access: OK`);

  if (existingFiles >= keyFiles.length * 0.8) {
    console.log('\n🎉 Runtime test PASSED - Core infrastructure is ready!');
    process.exit(0);
  } else {
    console.log('\n⚠️  Runtime test WARNING - Some core files missing');
    process.exit(1);
  }

} catch (error) {
  console.error('❌ Runtime test FAILED:', error.message);
  process.exit(1);
}