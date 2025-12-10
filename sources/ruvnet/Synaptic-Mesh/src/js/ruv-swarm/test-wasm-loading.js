import { default as WasmModuleLoader } from './src/wasm-loader.js';

async function testWasmLoading() {
  try {
    console.log('🔧 Testing WASM loading...\n');
    
    const loader = new WasmModuleLoader();
    await loader.initialize('progressive');
    
    console.log('✅ Loader initialized\n');
    
    const core = await loader.loadModule('core');
    
    const status = {
      hasExports: !!core.exports,
      exportKeys: core.exports ? Object.keys(core.exports).filter(k => !k.startsWith('__')).slice(0, 10) : [],
      isPlaceholder: core.isPlaceholder || false,
      memorySize: core.memory ? core.memory.buffer.byteLength : 0,
      hasWasmFunctions: core.exports && typeof core.exports.create_swarm_orchestrator === 'function'
    };
    
    console.log('📊 WASM Module Status:');
    console.log('- Has exports:', status.hasExports);
    console.log('- Is placeholder:', status.isPlaceholder);
    console.log('- Memory allocated:', status.memorySize, 'bytes');
    console.log('- Has WASM functions:', status.hasWasmFunctions);
    console.log('\n🔍 Available exports:', status.exportKeys.join(', '));
    
    // Test a WASM function
    if (core.exports && core.exports.create_swarm_orchestrator) {
      console.log('\n🧪 Testing WASM function...');
      try {
        const result = core.exports.create_swarm_orchestrator('mesh');
        console.log('✅ WASM function executed successfully:', result);
      } catch (e) {
        console.log('❌ WASM function error:', e.message);
      }
    }
    
    console.log('\n✨ WASM loading test complete!');
    
  } catch (error) {
    console.error('❌ WASM loading failed:', error);
    console.error('Stack:', error.stack);
  }
}

testWasmLoading();