#!/usr/bin/env node

// Test SPARC modes loading
import { loadSparcModes } from './dist/mcp/sparc-modes.js';

async function test() {
  console.log('🧪 Testing SPARC modes loading...\n');
  
  try {
    const modes = await loadSparcModes();
    console.log(`✅ Loaded ${modes.length} SPARC modes:\n`);
    
    modes.forEach((mode, index) => {
      console.log(`${index + 1}. ${mode.name}`);
      console.log(`   Description: ${mode.description}`);
      console.log(`   Tools: ${mode.tools.join(', ')}`);
      console.log('');
    });
  } catch (error) {
    console.error('❌ Error:', error);
  }
}

test();