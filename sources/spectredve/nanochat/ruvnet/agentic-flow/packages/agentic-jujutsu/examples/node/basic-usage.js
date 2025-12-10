#!/usr/bin/env node
/**
 * Basic usage example for Node.js
 */

const jj = require('../../pkg/node');

async function main() {
  console.log('\n🚀 agentic-jujutsu Node.js Example\n');
  
  try {
    console.log('📦 Package loaded successfully!');
    console.log('Available exports:', Object.keys(jj).join(', '));
    
    console.log('\n✅ Ready to use agentic-jujutsu in your Node.js application!\n');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

main();
