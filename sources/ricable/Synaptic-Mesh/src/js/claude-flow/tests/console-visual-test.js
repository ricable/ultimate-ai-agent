/**
 * Visual Test for Console Terminal
 * Captures screenshots of different console states
 */

import puppeteer from 'puppeteer';
import fs from 'fs/promises';
import path from 'path';

async function captureConsoleStates() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  
  // Create screenshots directory
  const screenshotDir = path.join(process.cwd(), 'tests/console-screenshots');
  await fs.mkdir(screenshotDir, { recursive: true });
  
  try {
    // 1. Initial load state
    await page.goto('http://localhost:3002/console/', {
      waitUntil: 'networkidle2'
    });
    await page.waitForSelector('.console-input', { timeout: 10000 });
    await page.screenshot({ 
      path: path.join(screenshotDir, '01-initial-load.png'),
      fullPage: true 
    });
    console.log('✅ Captured initial load state');
    
    // 2. After typing help command
    await page.type('.console-input', 'help');
    await page.screenshot({ 
      path: path.join(screenshotDir, '02-help-typed.png'),
      fullPage: true 
    });
    await page.keyboard.press('Enter');
    await new Promise(r => setTimeout(r, 1000));
    await page.screenshot({ 
      path: path.join(screenshotDir, '03-help-executed.png'),
      fullPage: true 
    });
    console.log('✅ Captured help command');
    
    // 3. Status command
    await page.type('.console-input', 'status');
    await page.keyboard.press('Enter');
    await new Promise(r => setTimeout(r, 1000));
    await page.screenshot({ 
      path: path.join(screenshotDir, '04-status-output.png'),
      fullPage: true 
    });
    console.log('✅ Captured status command');
    
    // 4. Multiple commands for scrolling
    for (let i = 0; i < 10; i++) {
      await page.type('.console-input', `echo Test line ${i}`);
      await page.keyboard.press('Enter');
      await new Promise(r => setTimeout(r, 100));
    }
    await page.screenshot({ 
      path: path.join(screenshotDir, '05-scrolled-content.png'),
      fullPage: true 
    });
    console.log('✅ Captured scrolled content');
    
    // 5. Clear command
    await page.type('.console-input', 'clear');
    await page.keyboard.press('Enter');
    await new Promise(r => setTimeout(r, 500));
    await page.screenshot({ 
      path: path.join(screenshotDir, '06-after-clear.png'),
      fullPage: true 
    });
    console.log('✅ Captured clear command result');
    
    // 6. Test error state
    await page.type('.console-input', 'invalidcommand123');
    await page.keyboard.press('Enter');
    await new Promise(r => setTimeout(r, 500));
    await page.screenshot({ 
      path: path.join(screenshotDir, '07-error-state.png'),
      fullPage: true 
    });
    console.log('✅ Captured error state');
    
    console.log('\n📸 All screenshots captured successfully!');
    console.log(`📁 Screenshots saved to: ${screenshotDir}`);
    
  } catch (error) {
    console.error('❌ Error capturing screenshots:', error);
  } finally {
    await browser.close();
  }
}

// Run the visual test
captureConsoleStates();