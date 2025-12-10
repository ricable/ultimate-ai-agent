#!/usr/bin/env node

// Direct test of .roomodes parsing
import fs from 'fs/promises';

async function test() {
  console.log('🧪 Testing .roomodes file parsing...\n');
  
  try {
    // Read the file
    const content = await fs.readFile('.roomodes', 'utf-8');
    console.log('✅ File read successfully');
    console.log(`📏 File size: ${content.length} bytes\n`);
    
    // Parse JSON
    const modesData = JSON.parse(content);
    console.log(`✅ JSON parsed successfully`);
    console.log(`📊 Number of modes: ${Object.keys(modesData).length}\n`);
    
    // List modes
    console.log('📋 Available modes:');
    Object.entries(modesData).forEach(([name, config], index) => {
      console.log(`${index + 1}. ${name}`);
      console.log(`   Description: ${config.description}`);
      console.log(`   Tools: ${config.tools.join(', ')}`);
    });
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

test();