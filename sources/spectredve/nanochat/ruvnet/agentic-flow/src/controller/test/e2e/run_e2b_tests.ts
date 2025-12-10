#!/usr/bin/env ts-node
/**
 * E2B Test Runner for Kubernetes Controller
 *
 * Usage:
 *   npm run test:e2b
 *   E2B_API_KEY=your_key npm run test:e2b
 */

import { runTests } from './e2b_kubernetes_test';
import * as fs from 'fs';
import * as path from 'path';

async function main() {
  console.log('╔═══════════════════════════════════════════════════════╗');
  console.log('║  Agentic-Jujutsu Kubernetes Controller E2B Tests     ║');
  console.log('╚═══════════════════════════════════════════════════════╝\n');

  // Check for E2B API key
  const apiKey = process.env.E2B_API_KEY;
  if (!apiKey) {
    console.error('❌ Error: E2B_API_KEY environment variable not set');
    console.error('   Please set your E2B API key:');
    console.error('   export E2B_API_KEY=your_api_key_here\n');
    process.exit(1);
  }

  console.log('✓ E2B API key found');
  console.log(`✓ API key: ${apiKey.substring(0, 10)}...${apiKey.substring(apiKey.length - 4)}\n`);

  try {
    // Run the tests
    const results = await runTests();

    // Generate test report
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        total: results.length,
        passed: results.filter(r => r.passed).length,
        failed: results.filter(r => !r.passed).length,
        successRate: ((results.filter(r => r.passed).length / results.length) * 100).toFixed(1),
      },
      results: results.map(r => ({
        name: r.name,
        passed: r.passed,
        duration: r.duration,
        error: r.error,
        logs: r.logs,
      })),
    };

    // Save report
    const reportDir = path.join(__dirname, '../../../test-reports');
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    const reportPath = path.join(reportDir, `e2b-test-${Date.now()}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 Test report saved to: ${reportPath}`);

    // Exit with appropriate code
    const exitCode = report.summary.failed > 0 ? 1 : 0;
    console.log(`\n${exitCode === 0 ? '✅ All tests passed!' : '❌ Some tests failed'}`);
    process.exit(exitCode);
  } catch (error) {
    console.error('\n❌ Fatal error running tests:');
    console.error(error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}
