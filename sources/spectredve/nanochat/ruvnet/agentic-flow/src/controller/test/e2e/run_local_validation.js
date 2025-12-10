#!/usr/bin/env node
/**
 * Local Validation Test Runner for Kubernetes Controller
 * Validates implementation files before E2B deployment
 */

const fs = require('fs');
const path = require('path');

// Base path relative to this script
const BASE_PATH = path.join(__dirname, '../../..');
const CHARTS_PATH = path.join(__dirname, '../../../../../charts');

console.log('╔══════════════════════════════════════════════════════════╗');
console.log('║  Kubernetes Controller Local Validation Tests           ║');
console.log('╚══════════════════════════════════════════════════════════╝\n');

// Check for E2B API key
const apiKey = process.env.E2B_API_KEY;
if (apiKey) {
  console.log('✓ E2B API key found');
  console.log(`✓ Key: ${apiKey.substring(0, 10)}...${apiKey.substring(apiKey.length - 4)}\n`);
} else {
  console.log('⚠ E2B API key not found (optional for local validation)\n');
}

const results = [];
const startTime = Date.now();

/**
 * Test controller source files
 */
function testControllerSource() {
  console.log('🧪 Test 1: Controller Source Files');
  const testStart = Date.now();

  const files = [
    'api/v1/application_types.go',
    'api/v1/cluster_types.go',
    'api/v1/groupversion_info.go',
    'cmd/manager/main.go',
    'internal/controller/application_controller.go',
    'internal/jujutsu/client.go',
    'internal/policy/validator.go',
    'internal/cluster/manager.go',
    'go.mod',
    'Makefile',
    'Dockerfile',
    'README.md',
  ];

  let allExist = true;
  let stats = { total: 0, found: 0, missing: 0, size: 0 };

  for (const file of files) {
    const fullPath = path.join(BASE_PATH, file);
    stats.total++;

    if (fs.existsSync(fullPath)) {
      const fileStats = fs.statSync(fullPath);
      stats.found++;
      stats.size += fileStats.size;
      console.log(`   ✓ ${file} (${fileStats.size} bytes)`);
    } else {
      stats.missing++;
      allExist = false;
      console.log(`   ❌ Missing: ${file}`);
    }
  }

  const duration = Date.now() - testStart;
  console.log(`   Summary: ${stats.found}/${stats.total} files, ${stats.size} bytes total`);

  if (allExist) {
    console.log(`✅ Controller Source PASSED (${duration}ms)\n`);
    results.push({ name: 'Controller Source', passed: true, duration, stats });
  } else {
    console.log(`❌ Controller Source FAILED (${duration}ms)\n`);
    results.push({ name: 'Controller Source', passed: false, duration, stats });
  }
}

/**
 * Test Go code structure
 */
function testGoCodeStructure() {
  console.log('🧪 Test 2: Go Code Structure');
  const testStart = Date.now();

  try {
    // Check Application CRD
    const appTypesPath = path.join(BASE_PATH, 'api/v1/application_types.go');
    if (!fs.existsSync(appTypesPath)) {
      throw new Error('Application types file not found');
    }

    const appContent = fs.readFileSync(appTypesPath, 'utf8');
    const checks = [
      { pattern: /type ApplicationSpec struct/, name: 'ApplicationSpec' },
      { pattern: /type ApplicationStatus struct/, name: 'ApplicationStatus' },
      { pattern: /type Application struct/, name: 'Application' },
      { pattern: /ProgressiveDelivery/, name: 'ProgressiveDelivery' },
      { pattern: /Infrastructure/, name: 'Infrastructure' },
    ];

    let allFound = true;
    for (const check of checks) {
      if (check.pattern.test(appContent)) {
        console.log(`   ✓ Found: ${check.name}`);
      } else {
        console.log(`   ❌ Missing: ${check.name}`);
        allFound = false;
      }
    }

    const lines = appContent.split('\n').length;
    console.log(`   ℹ Application types: ${lines} lines`);

    const duration = Date.now() - testStart;
    if (allFound) {
      console.log(`✅ Go Code Structure PASSED (${duration}ms)\n`);
      results.push({ name: 'Go Code Structure', passed: true, duration });
    } else {
      console.log(`❌ Go Code Structure FAILED (${duration}ms)\n`);
      results.push({ name: 'Go Code Structure', passed: false, duration });
    }
  } catch (error) {
    const duration = Date.now() - testStart;
    console.log(`❌ Go Code Structure ERROR: ${error.message} (${duration}ms)\n`);
    results.push({ name: 'Go Code Structure', passed: false, duration, error: error.message });
  }
}

/**
 * Test Helm chart files
 */
function testHelmChart() {
  console.log('🧪 Test 3: Helm Chart Files');
  const testStart = Date.now();

  const chartPath = path.join(CHARTS_PATH, 'jujutsu-gitops-controller');
  const files = [
    'Chart.yaml',
    'values.yaml',
    'templates/deployment.yaml',
    'templates/_helpers.tpl',
  ];

  let allExist = true;
  let stats = { total: 0, found: 0, missing: 0 };

  for (const file of files) {
    const fullPath = path.join(chartPath, file);
    stats.total++;

    if (fs.existsSync(fullPath)) {
      stats.found++;
      const fileStats = fs.statSync(fullPath);
      console.log(`   ✓ ${file} (${fileStats.size} bytes)`);
    } else {
      stats.missing++;
      allExist = false;
      console.log(`   ❌ Missing: ${file}`);
    }
  }

  const duration = Date.now() - testStart;
  console.log(`   Summary: ${stats.found}/${stats.total} chart files`);

  if (allExist) {
    console.log(`✅ Helm Chart PASSED (${duration}ms)\n`);
    results.push({ name: 'Helm Chart', passed: true, duration, stats });
  } else {
    console.log(`❌ Helm Chart FAILED (${duration}ms)\n`);
    results.push({ name: 'Helm Chart', passed: false, duration, stats });
  }
}

/**
 * Test test files
 */
function testTestFiles() {
  console.log('🧪 Test 4: Test Files');
  const testStart = Date.now();

  const testFiles = [
    'test/unit/jujutsu_test.go',
    'test/unit/policy_test.go',
    'test/performance/benchmark_test.go',
    'test/e2e/e2b_kubernetes_test.ts',
    'test/e2e/package.json',
  ];

  let allExist = true;
  let stats = { total: 0, found: 0, missing: 0 };

  for (const file of testFiles) {
    const fullPath = path.join(BASE_PATH, file);
    stats.total++;

    if (fs.existsSync(fullPath)) {
      stats.found++;
      const fileStats = fs.statSync(fullPath);
      console.log(`   ✓ ${file} (${fileStats.size} bytes)`);
    } else {
      stats.missing++;
      allExist = false;
      console.log(`   ❌ Missing: ${file}`);
    }
  }

  const duration = Date.now() - testStart;
  console.log(`   Summary: ${stats.found}/${stats.total} test files`);

  if (allExist) {
    console.log(`✅ Test Files PASSED (${duration}ms)\n`);
    results.push({ name: 'Test Files', passed: true, duration, stats });
  } else {
    console.log(`❌ Test Files FAILED (${duration}ms)\n`);
    results.push({ name: 'Test Files', passed: false, duration, stats });
  }
}

/**
 * Test documentation
 */
function testDocumentation() {
  console.log('🧪 Test 5: Documentation');
  const testStart = Date.now();

  const docFiles = [
    'README.md',
    '../../../docs/kubernetes-implementation-summary.md',
    '../../../docs/specs/agentic-jujutsu-spec.md',
  ];

  let allExist = true;
  let stats = { total: 0, found: 0, missing: 0, totalLines: 0 };

  for (const file of docFiles) {
    const fullPath = path.join(BASE_PATH, file);
    stats.total++;

    if (fs.existsSync(fullPath)) {
      stats.found++;
      const content = fs.readFileSync(fullPath, 'utf8');
      const lines = content.split('\n').length;
      stats.totalLines += lines;
      console.log(`   ✓ ${path.basename(file)} (${lines} lines)`);
    } else {
      stats.missing++;
      allExist = false;
      console.log(`   ❌ Missing: ${file}`);
    }
  }

  const duration = Date.now() - testStart;
  console.log(`   Summary: ${stats.found}/${stats.total} docs, ${stats.totalLines} lines total`);

  if (allExist) {
    console.log(`✅ Documentation PASSED (${duration}ms)\n`);
    results.push({ name: 'Documentation', passed: true, duration, stats });
  } else {
    console.log(`❌ Documentation FAILED (${duration}ms)\n`);
    results.push({ name: 'Documentation', passed: false, duration, stats });
  }
}

// Run all tests
testControllerSource();
testGoCodeStructure();
testHelmChart();
testTestFiles();
testDocumentation();

// Generate summary
const totalDuration = Date.now() - startTime;
const passed = results.filter(r => r.passed).length;
const failed = results.filter(r => !r.passed).length;
const successRate = ((passed / results.length) * 100).toFixed(1);

console.log('═══════════════════════════════════════════════════════════');
console.log('📊 TEST SUMMARY');
console.log('═══════════════════════════════════════════════════════════');
console.log(`Total Tests:    ${results.length}`);
console.log(`✅ Passed:      ${passed}`);
console.log(`❌ Failed:      ${failed}`);
console.log(`Success Rate:   ${successRate}%`);
console.log(`Total Duration: ${totalDuration}ms`);
console.log('═══════════════════════════════════════════════════════════\n');

// Detailed results
if (passed === results.length) {
  console.log('🎉 ALL TESTS PASSED - Ready for E2B deployment!\n');
} else {
  console.log('⚠ Some tests failed - Review before E2B deployment\n');
}

// Save report
const report = {
  timestamp: new Date().toISOString(),
  environment: 'local-validation',
  e2bApiKey: apiKey ? `${apiKey.substring(0, 10)}...${apiKey.substring(apiKey.length - 4)}` : 'not-provided',
  summary: {
    total: results.length,
    passed,
    failed,
    successRate: parseFloat(successRate),
    duration: totalDuration,
  },
  results,
};

const reportDir = path.join(BASE_PATH, '../../../test-reports');
if (!fs.existsSync(reportDir)) {
  fs.mkdirSync(reportDir, { recursive: true });
}

const reportPath = path.join(reportDir, `local-validation-${Date.now()}.json`);
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
console.log(`📄 Test report saved: ${reportPath}\n`);

// List all implementation files
console.log('📁 Implementation Files:');
console.log('═══════════════════════════════════════════════════════════');

function listFiles(dir, prefix = '') {
  try {
    const items = fs.readdirSync(dir);
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stats = fs.statSync(fullPath);
      if (stats.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
        console.log(`${prefix}📁 ${item}/`);
        listFiles(fullPath, prefix + '  ');
      } else if (stats.isFile()) {
        console.log(`${prefix}📄 ${item} (${stats.size} bytes)`);
      }
    }
  } catch (error) {
    // Skip directories we can't read
  }
}

console.log('\nController Implementation:');
listFiles(BASE_PATH);

console.log('\nHelm Chart:');
listFiles(path.join(CHARTS_PATH, 'jujutsu-gitops-controller'));

// Exit code
process.exit(failed > 0 ? 1 : 0);
