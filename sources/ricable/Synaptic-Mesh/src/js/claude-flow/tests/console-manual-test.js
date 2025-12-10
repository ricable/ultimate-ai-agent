/**
 * Manual Console Terminal Test
 * Quick test to check console functionality
 */

import puppeteer from 'puppeteer';

async function testConsole() {
  const browser = await puppeteer.launch({
    headless: false, // Show browser window
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
    devtools: true
  });
  
  const page = await browser.newPage();
  await page.goto('http://localhost:3002/console/', {
    waitUntil: 'networkidle2'
  });
  
  // Wait for console to load
  await page.waitForSelector('.console-input', { timeout: 10000 });
  
  console.log('Console loaded! Browser window is open for manual testing.');
  console.log('Check the following:');
  console.log('1. Type "help" and see if command list appears');
  console.log('2. Type "clear" to clear console');
  console.log('3. Type "status" to see system status');
  console.log('4. Test UP/DOWN arrows for history');
  console.log('5. Test Ctrl+L to clear');
  console.log('6. Check if timestamps are visible');
  console.log('\nPress Ctrl+C to close when done testing...');
  
  // Keep browser open
  await new Promise(() => {});
}

testConsole();