import { DefaultClaudeDetector, DefaultMCPConfigurator, MCPServerConfig, MCPConfig } from './src/onboarding/index.js';
import { existsSync } from 'fs';

console.log('🧪 Testing PR #34 - Comprehensive Onboarding Integration\n');

async function testPR34() {
  let allTestsPassed = true;
  
  try {
    // Test 1: Check if onboarding module exists
    console.log('1️⃣ Testing onboarding module existence...');
    const onboardingPath = './src/onboarding/index.js';
    if (existsSync(onboardingPath)) {
      console.log('   ✅ Onboarding module found at:', onboardingPath);
    } else {
      console.log('   ❌ Onboarding module NOT found');
      allTestsPassed = false;
    }
    
    // Test 2: Test Claude detector instantiation
    console.log('\n2️⃣ Testing DefaultClaudeDetector...');
    try {
      const detector = new DefaultClaudeDetector();
      console.log('   ✅ DefaultClaudeDetector instantiated successfully');
      console.log('   📍 Search paths configured:', detector.searchPaths.length);
      
      // Test detection (won't find Claude in test env, but should not error)
      const claudeInfo = await detector.detect();
      console.log('   ✅ Detection method works (found:', claudeInfo.installed, ')');
    } catch (error) {
      console.log('   ❌ DefaultClaudeDetector failed:', error.message);
      allTestsPassed = false;
    }
    
    // Test 3: Test MCP configuration classes
    console.log('\n3️⃣ Testing MCP configuration classes...');
    try {
      // Test MCPServerConfig
      const serverConfig = new MCPServerConfig(
        'node',
        ['bin/ruv-swarm-clean.js', 'mcp', 'start'],
        {},
        true
      );
      console.log('   ✅ MCPServerConfig created successfully');
      
      // Test MCPConfig
      const mcpConfig = new MCPConfig({
        'ruv-swarm': serverConfig
      }, true, true);
      console.log('   ✅ MCPConfig created successfully');
      console.log('   📍 Servers configured:', Object.keys(mcpConfig.servers).join(', '));
    } catch (error) {
      console.log('   ❌ MCP configuration classes failed:', error.message);
      allTestsPassed = false;
    }
    
    // Test 4: Test MCP configurator
    console.log('\n4️⃣ Testing DefaultMCPConfigurator...');
    try {
      const configurator = new DefaultMCPConfigurator();
      console.log('   ✅ DefaultMCPConfigurator instantiated successfully');
      
      // Test config generation (without requiring actual ruv-swarm availability)
      const config = configurator._generateRuvSwarmConfig();
      console.log('   ✅ Config generation works');
      console.log('   📍 Generated config has', Object.keys(config.servers).length, 'server(s)');
    } catch (error) {
      console.log('   ❌ DefaultMCPConfigurator failed:', error.message);
      allTestsPassed = false;
    }
    
    // Test 5: Check integration with existing code
    console.log('\n5️⃣ Testing integration with existing ruv-swarm...');
    try {
      // Check if the binary exists
      const binaryPath = 'bin/ruv-swarm-clean.js';
      if (existsSync(binaryPath)) {
        console.log('   ✅ ruv-swarm binary exists');
        
        // Import to check syntax
        await import('./bin/ruv-swarm-clean.js');
        console.log('   ✅ Binary imports successfully (no syntax errors)');
      } else {
        console.log('   ⚠️  Binary not found at expected location');
      }
    } catch (error) {
      // Import errors are expected since it's a CLI
      if (error.message.includes('process.argv')) {
        console.log('   ✅ Binary is a valid CLI script');
      } else {
        console.log('   ❌ Integration issue:', error.message);
        allTestsPassed = false;
      }
    }
    
    // Summary
    console.log('\n' + '='.repeat(60));
    if (allTestsPassed) {
      console.log('✅ ALL TESTS PASSED - PR #34 is ready for review!');
      console.log('\n📋 Summary:');
      console.log('   - Onboarding module properly structured');
      console.log('   - All classes instantiate correctly');
      console.log('   - No import/syntax errors detected');
      console.log('   - Integration points are clean');
    } else {
      console.log('❌ SOME TESTS FAILED - PR #34 needs attention');
      process.exit(1);
    }
    
  } catch (error) {
    console.error('\n💥 Unexpected error during testing:', error);
    console.error(error.stack);
    process.exit(1);
  }
}

// Run tests
testPR34();