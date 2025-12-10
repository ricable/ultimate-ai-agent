const { JJWrapper, JJConfig } = require('../index.js')
const assert = require('assert')

async function testBasicAPI() {
  console.log('Testing JJWrapper constructor...')
  const jj = new JJWrapper()
  assert(jj !== null, 'JJWrapper should be created')
  console.log('✅ Constructor works')

  console.log('Testing getConfig...')
  const config = jj.getConfig()
  assert(config.jj_path !== undefined, 'Config should have jj_path')
  console.log('✅ getConfig works')

  console.log('Testing getStats...')
  const stats = jj.getStats()
  assert(typeof stats === 'string', 'Stats should be string')
  console.log('✅ getStats works')
}

testBasicAPI().catch(err => {
  console.error('❌ Test failed:', err)
  process.exit(1)
})
