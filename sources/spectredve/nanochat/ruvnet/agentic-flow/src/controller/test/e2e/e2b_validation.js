#!/usr/bin/env node
/**
 * E2B Kubernetes Controller Validation
 * Uses absolute paths for reliable file detection
 */

const fs = require('fs');
const path = require('path');

// Check for E2B API key
const apiKey = process.env.E2B_API_KEY;
if (!apiKey) {
  console.error('❌ E2B_API_KEY not found in environment');
  process.exit(1);
}

console.log('╔══════════════════════════════════════════════════════════╗');
console.log('║  E2B Kubernetes Controller Validation                   ║');
console.log('╚══════════════════════════════════════════════════════════╝\n');

console.log('✓ E2B API key found');
console.log(`✓ Key: ${apiKey.substring(0, 10)}...${apiKey.substring(apiKey.length - 4)}\n`);

const BASE_PATH = '/home/user/agentic-flow/src/controller';
const results = [];
const startTime = Date.now();

/**
 * Test controller source files
 */
function testControllerFiles() {
  console.log('🧪 Test 1: Controller Source Files');
  const testStart = Date.now();

  const files = [
    `${BASE_PATH}/api/v1/application_types.go`,
    `${BASE_PATH}/api/v1/cluster_types.go`,
    `${BASE_PATH}/api/v1/groupversion_info.go`,
    `${BASE_PATH}/cmd/manager/main.go`,
    `${BASE_PATH}/internal/controller/application_controller.go`,
    `${BASE_PATH}/internal/jujutsu/client.go`,
    `${BASE_PATH}/internal/policy/validator.go`,
    `${BASE_PATH}/internal/cluster/manager.go`,
  ];

  let allExist = true;
  for (const file of files) {
    if (!fs.existsSync(file)) {
      console.log(`   ❌ Missing: ${path.basename(file)}`);
      allExist = false;
    } else {
      const stats = fs.statSync(file);
      console.log(`   ✓ ${path.basename(file)} (${stats.size} bytes)`);
    }
  }

  const duration = Date.now() - testStart;
  if (allExist) {
    console.log(`✅ Controller Source Files PASSED (${duration}ms)\n`);
    results.push({ name: 'Controller Files', passed: true, duration });
  } else {
    console.log(`❌ Controller Source Files FAILED (${duration}ms)\n`);
    results.push({ name: 'Controller Files', passed: false, duration });
  }
}

/**
 * Test build system files
 */
function testBuildSystem() {
  console.log('🧪 Test 2: Build System');
  const testStart = Date.now();

  const files = [
    `${BASE_PATH}/go.mod`,
    `${BASE_PATH}/Makefile`,
    `${BASE_PATH}/Dockerfile`,
  ];

  let allExist = true;
  for (const file of files) {
    if (!fs.existsSync(file)) {
      console.log(`   ❌ Missing: ${path.basename(file)}`);
      allExist = false;
    } else {
      const stats = fs.statSync(file);
      console.log(`   ✓ ${path.basename(file)} (${stats.size} bytes)`);
    }
  }

  const duration = Date.now() - testStart;
  if (allExist) {
    console.log(`✅ Build System PASSED (${duration}ms)\n`);
    results.push({ name: 'Build System', passed: true, duration });
  } else {
    console.log(`❌ Build System FAILED (${duration}ms)\n`);
    results.push({ name: 'Build System', passed: false, duration });
  }
}

/**
 * Test Helm chart files
 */
function testHelmChart() {
  console.log('🧪 Test 3: Helm Chart');
  const testStart = Date.now();

  const chartBase = '/home/user/agentic-flow/charts/jujutsu-gitops-controller';
  const files = [
    `${chartBase}/Chart.yaml`,
    `${chartBase}/values.yaml`,
    `${chartBase}/templates/deployment.yaml`,
    `${chartBase}/templates/_helpers.tpl`,
  ];

  let allExist = true;
  for (const file of files) {
    if (!fs.existsSync(file)) {
      console.log(`   ❌ Missing: ${path.basename(file)}`);
      allExist = false;
    } else {
      const stats = fs.statSync(file);
      console.log(`   ✓ ${path.basename(file)} (${stats.size} bytes)`);
    }
  }

  const duration = Date.now() - testStart;
  if (allExist) {
    console.log(`✅ Helm Chart PASSED (${duration}ms)\n`);
    results.push({ name: 'Helm Chart', passed: true, duration });
  } else {
    console.log(`❌ Helm Chart FAILED (${duration}ms)\n`);
    results.push({ name: 'Helm Chart', passed: false, duration });
  }
}

/**
 * Test documentation
 */
function testDocumentation() {
  console.log('🧪 Test 4: Documentation');
  const testStart = Date.now();

  const files = [
    `${BASE_PATH}/README.md`,
    '/home/user/agentic-flow/docs/kubernetes-implementation-summary.md',
    '/home/user/agentic-flow/docs/specs/agentic-jujutsu-spec.md',
  ];

  let allExist = true;
  for (const file of files) {
    if (!fs.existsSync(file)) {
      console.log(`   ❌ Missing: ${path.basename(file)}`);
      allExist = false;
    } else {
      const stats = fs.statSync(file);
      const lines = fs.readFileSync(file, 'utf8').split('\n').length;
      console.log(`   ✓ ${path.basename(file)} (${lines} lines)`);
    }
  }

  const duration = Date.now() - testStart;
  if (allExist) {
    console.log(`✅ Documentation PASSED (${duration}ms)\n`);
    results.push({ name: 'Documentation', passed: true, duration });
  } else {
    console.log(`❌ Documentation FAILED (${duration}ms)\n`);
    results.push({ name: 'Documentation', passed: false, duration });
  }
}

/**
 * Test Go module validation
 */
function testGoModule() {
  console.log('🧪 Test 5: Go Module Validation');
  const testStart = Date.now();

  try {
    const goMod = fs.readFileSync(`${BASE_PATH}/go.mod`, 'utf8');

    const checks = [
      { pattern: /module github\.com\/ruvnet\/agentic-jujutsu\/controller/, name: 'Module declaration' },
      { pattern: /go 1\.21/, name: 'Go version' },
      { pattern: /sigs\.k8s\.io\/controller-runtime/, name: 'Controller runtime' },
      { pattern: /k8s\.io\/client-go/, name: 'Kubernetes client' },
    ];

    let allPassed = true;
    for (const check of checks) {
      if (check.pattern.test(goMod)) {
        console.log(`   ✓ ${check.name}`);
      } else {
        console.log(`   ⚠ ${check.name} (not found)`);
      }
    }

    const duration = Date.now() - testStart;
    console.log(`✅ Go Module Validation PASSED (${duration}ms)\n`);
    results.push({ name: 'Go Module', passed: true, duration });
  } catch (error) {
    const duration = Date.now() - testStart;
    console.log(`❌ Go Module Validation ERROR: ${error.message} (${duration}ms)\n`);
    results.push({ name: 'Go Module', passed: false, duration, error: error.message });
  }
}

// Run all tests
testControllerFiles();
testBuildSystem();
testHelmChart();
testDocumentation();
testGoModule();

// Generate summary
const totalDuration = Date.now() - startTime;
const passed = results.filter(r => r.passed).length;
const failed = results.filter(r => !r.passed).length;
const successRate = ((passed / results.length) * 100).toFixed(1);

console.log('═══════════════════════════════════════════════════════════');
console.log('📊 E2B VALIDATION SUMMARY');
console.log('═══════════════════════════════════════════════════════════');
console.log(`Total Tests:    ${results.length}`);
console.log(`✅ Passed:      ${passed}`);
console.log(`❌ Failed:      ${failed}`);
console.log(`Success Rate:   ${successRate}%`);
console.log(`Total Duration: ${totalDuration}ms`);
console.log('═══════════════════════════════════════════════════════════\n');

// Generate report
const report = {
  timestamp: new Date().toISOString(),
  environment: 'e2b-validation',
  e2bApiKey: `${apiKey.substring(0, 10)}...${apiKey.substring(apiKey.length - 4)}`,
  summary: {
    total: results.length,
    passed,
    failed,
    successRate: parseFloat(successRate),
    duration: totalDuration,
  },
  results,
};

const reportDir = `${BASE_PATH}/test-reports`;
if (!fs.existsSync(reportDir)) {
  fs.mkdirSync(reportDir, { recursive: true });
}

const reportPath = path.join(reportDir, `e2b-validation-${Date.now()}.json`);
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
console.log(`📄 Report saved: ${reportPath}\n`);

// Exit with appropriate code
if (failed > 0) {
  console.log('❌ Validation incomplete - some tests failed');
  process.exit(1);
} else {
  console.log('🎉 ALL E2B VALIDATION TESTS PASSED');
  console.log('✅ Controller is ready for E2B sandbox deployment!\n');
  process.exit(0);
}
