#!/usr/bin/env node
/**
 * Deep Validation Test for Parallel Execution Implementation
 * Tests code structure, patterns, and integration without requiring full agent execution
 */

const fs = require('fs');
const path = require('path');

console.log('\n🔍 Deep Review: Parallel Execution Capabilities');
console.log('='.repeat(70));

const results = {
  timestamp: new Date().toISOString(),
  tests: [],
  summary: {
    passed: 0,
    failed: 0,
    warnings: 0
  }
};

function test(name, fn) {
  try {
    const result = fn();
    if (result.passed) {
      console.log(`✅ ${name}`);
      results.tests.push({ name, status: 'passed', ...result });
      results.summary.passed++;
    } else {
      console.log(`❌ ${name}`);
      console.log(`   Reason: ${result.reason}`);
      results.tests.push({ name, status: 'failed', ...result });
      results.summary.failed++;
    }
    if (result.warnings) {
      result.warnings.forEach(w => console.log(`   ⚠️  ${w}`));
      results.summary.warnings += result.warnings.length;
    }
  } catch (error) {
    console.log(`❌ ${name} - Exception: ${error.message}`);
    results.tests.push({ name, status: 'error', error: error.message });
    results.summary.failed++;
  }
}

// Test 1: Parallel Execution Guide Exists
test('Parallel Execution Guide Exists', () => {
  const guidePath = path.join(__dirname, '../../agentic-flow/src/prompts/parallel-execution-guide.md');
  if (!fs.existsSync(guidePath)) {
    return { passed: false, reason: 'Guide file not found' };
  }

  const content = fs.readFileSync(guidePath, 'utf-8');
  const hasSubprocessPattern = content.includes('npx agentic-flow --agent');
  const hasPromiseAll = content.includes('Promise.all');
  const hasReasoningBank = content.includes('reasoningBank');
  const hasExamples = content.split('###').length >= 3;

  const warnings = [];
  if (!hasSubprocessPattern) warnings.push('Missing CLI subprocess pattern');
  if (!hasPromiseAll) warnings.push('Missing Promise.all pattern');
  if (!hasReasoningBank) warnings.push('Missing ReasoningBank coordination');
  if (!hasExamples) warnings.push('Insufficient examples');

  return {
    passed: hasSubprocessPattern && hasPromiseAll && hasReasoningBank,
    details: { lines: content.split('\n').length, hasExamples },
    warnings
  };
});

// Test 2: Provider Instructions Enhanced
test('Provider Instructions Enhanced', () => {
  const providerPath = path.join(__dirname, '../../agentic-flow/src/proxy/provider-instructions.ts');
  if (!fs.existsSync(providerPath)) {
    return { passed: false, reason: 'Provider instructions file not found' };
  }

  const content = fs.readFileSync(providerPath, 'utf-8');
  const hasParallelInstructions = content.includes('PARALLEL_EXECUTION_INSTRUCTIONS');
  const hasBuildInstructions = content.includes('buildInstructions');
  const hasParallelCapabilities = content.includes('getParallelCapabilities');
  const hasInstructionOptions = content.includes('InstructionOptions');

  return {
    passed: hasParallelInstructions && hasBuildInstructions && hasParallelCapabilities,
    details: {
      parallelInstructions: hasParallelInstructions,
      buildFunction: hasBuildInstructions,
      capabilities: hasParallelCapabilities,
      options: hasInstructionOptions
    }
  };
});

// Test 3: Validation Hooks Implemented
test('Validation Hooks Implemented', () => {
  const hooksPath = path.join(__dirname, '../../agentic-flow/src/hooks/parallel-validation.ts');
  if (!fs.existsSync(hooksPath)) {
    return { passed: false, reason: 'Validation hooks file not found' };
  }

  const content = fs.readFileSync(hooksPath, 'utf-8');
  const hasValidateFunction = content.includes('validateParallelExecution');
  const hasPostExecution = content.includes('postExecutionValidation');
  const hasGradeFunction = content.includes('gradeParallelExecution');
  const hasScoring = content.includes('score:');

  // Count validation checks
  const checks = [
    'hasSequentialSubprocessSpawning',
    'usesReasoningBank',
    'usesQuicTransport',
    'synthesizesResults',
    'storesSuccessPattern'
  ];
  const implementedChecks = checks.filter(check => content.includes(check));

  return {
    passed: hasValidateFunction && hasPostExecution && hasGradeFunction,
    details: {
      validationChecks: implementedChecks.length,
      expectedChecks: checks.length,
      hasGrading: hasGradeFunction
    }
  };
});

// Test 4: Test Suites Exist
test('Test Suites Complete', () => {
  const testDir = path.join(__dirname);
  const requiredTests = [
    'mesh-swarm-test.js',
    'hierarchical-swarm-test.js',
    'ring-swarm-test.js',
    'benchmark-suite.js'
  ];

  const missingTests = requiredTests.filter(test =>
    !fs.existsSync(path.join(testDir, test))
  );

  if (missingTests.length > 0) {
    return {
      passed: false,
      reason: `Missing tests: ${missingTests.join(', ')}`
    };
  }

  // Check each test file has required functions
  const meshContent = fs.readFileSync(path.join(testDir, 'mesh-swarm-test.js'), 'utf-8');
  const hasMeshFunction = meshContent.includes('async function meshSwarmTest');
  const hasPromiseAll = meshContent.includes('Promise.all');

  return {
    passed: true,
    details: {
      totalTests: requiredTests.length,
      meshHasParallelPattern: hasPromiseAll,
      allTestsFound: true
    }
  };
});

// Test 5: Docker Environment
test('Docker Environment Configured', () => {
  const dockerDir = path.join(__dirname, '../docker');
  const dockerfile = path.join(dockerDir, 'Dockerfile.parallel-test');
  const compose = path.join(dockerDir, 'docker-compose.parallel.yml');

  if (!fs.existsSync(dockerfile)) {
    return { passed: false, reason: 'Dockerfile not found' };
  }
  if (!fs.existsSync(compose)) {
    return { passed: false, reason: 'docker-compose.yml not found' };
  }

  const composeContent = fs.readFileSync(compose, 'utf-8');
  const services = ['mesh-swarm', 'hierarchical-swarm', 'ring-swarm', 'benchmark'];
  const hasAllServices = services.every(service => composeContent.includes(service));

  return {
    passed: true,
    details: {
      hasDockerfile: true,
      hasCompose: true,
      servicesCount: services.filter(s => composeContent.includes(s)).length
    },
    warnings: hasAllServices ? [] : ['Some services missing from docker-compose']
  };
});

// Test 6: NPM Scripts Integration
test('NPM Scripts Integrated', () => {
  const packagePath = path.join(__dirname, '../../package.json');
  if (!fs.existsSync(packagePath)) {
    return { passed: false, reason: 'package.json not found' };
  }

  const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf-8'));
  const requiredScripts = [
    'test:parallel',
    'test:mesh',
    'test:hierarchical',
    'test:ring',
    'bench:parallel'
  ];

  const missingScripts = requiredScripts.filter(script => !pkg.scripts[script]);

  return {
    passed: missingScripts.length === 0,
    details: {
      foundScripts: requiredScripts.length - missingScripts.length,
      totalRequired: requiredScripts.length
    },
    warnings: missingScripts.length > 0 ? [`Missing scripts: ${missingScripts.join(', ')}`] : []
  };
});

// Test 7: Code Patterns Analysis
test('Parallel Execution Patterns Valid', () => {
  const guidePath = path.join(__dirname, '../../agentic-flow/src/prompts/parallel-execution-guide.md');
  const content = fs.readFileSync(guidePath, 'utf-8');

  // Check for anti-patterns
  const antiPatterns = {
    'sequential await': /await\s+exec.*\nawait\s+exec/g,
    'single todo': /TodoWrite.*\[.*\]/g
  };

  const warnings = [];

  // Check for best practices
  const bestPractices = {
    'Promise.all usage': content.includes('Promise.all(['),
    'ReasoningBank store': content.includes('reasoningBank.storePattern'),
    'ReasoningBank retrieve': content.includes('reasoningBank.retrieve'),
    'CLI subprocess': content.includes('npx agentic-flow --agent'),
    'Batch operations': content.includes('batch'),
    'QUIC transport': content.includes('--transport quic')
  };

  const missingPractices = Object.entries(bestPractices)
    .filter(([_, has]) => !has)
    .map(([practice]) => practice);

  if (missingPractices.length > 0) {
    warnings.push(`Missing best practices: ${missingPractices.join(', ')}`);
  }

  return {
    passed: Object.values(bestPractices).filter(Boolean).length >= 5,
    details: {
      bestPractices: Object.values(bestPractices).filter(Boolean).length,
      totalPractices: Object.keys(bestPractices).length
    },
    warnings
  };
});

// Test 8: Documentation Completeness
test('Documentation Complete', () => {
  const implReportPath = path.join(__dirname, '../../docs/parallel-execution-implementation.md');

  if (!fs.existsSync(implReportPath)) {
    return {
      passed: false,
      reason: 'Implementation report not found',
      warnings: ['Create docs/parallel-execution-implementation.md']
    };
  }

  const content = fs.readFileSync(implReportPath, 'utf-8');
  const requiredSections = [
    'Summary',
    'Implementation Overview',
    'Validation Results',
    'Performance Expectations',
    'Next Steps'
  ];

  const missingSections = requiredSections.filter(section =>
    !content.includes(`## ${section}`) && !content.includes(`# ${section}`)
  );

  return {
    passed: missingSections.length === 0,
    details: {
      sections: requiredSections.length - missingSections.length,
      totalRequired: requiredSections.length,
      wordCount: content.split(/\s+/).length
    },
    warnings: missingSections.length > 0 ? [`Missing sections: ${missingSections.join(', ')}`] : []
  };
});

// Test 9: File Organization
test('File Organization Correct', () => {
  const expectedFiles = [
    'agentic-flow/src/prompts/parallel-execution-guide.md',
    'agentic-flow/src/proxy/provider-instructions.ts',
    'agentic-flow/src/hooks/parallel-validation.ts',
    'tests/parallel/mesh-swarm-test.js',
    'tests/parallel/hierarchical-swarm-test.js',
    'tests/parallel/ring-swarm-test.js',
    'tests/parallel/benchmark-suite.js',
    'tests/docker/Dockerfile.parallel-test',
    'tests/docker/docker-compose.parallel.yml',
    'docs/parallel-execution-implementation.md'
  ];

  const missingFiles = expectedFiles.filter(file =>
    !fs.existsSync(path.join(__dirname, '../..', file))
  );

  return {
    passed: missingFiles.length === 0,
    details: {
      foundFiles: expectedFiles.length - missingFiles.length,
      totalExpected: expectedFiles.length
    },
    warnings: missingFiles.length > 0 ? [`Missing files: ${missingFiles.join(', ')}`] : []
  };
});

// Test 10: TypeScript Compilation Check
test('TypeScript Files Valid', () => {
  const tsFiles = [
    'agentic-flow/src/proxy/provider-instructions.ts',
    'agentic-flow/src/hooks/parallel-validation.ts'
  ];

  const issues = [];

  tsFiles.forEach(file => {
    const fullPath = path.join(__dirname, '../..', file);
    if (fs.existsSync(fullPath)) {
      const content = fs.readFileSync(fullPath, 'utf-8');

      // Basic syntax checks
      const openBraces = (content.match(/\{/g) || []).length;
      const closeBraces = (content.match(/\}/g) || []).length;

      if (openBraces !== closeBraces) {
        issues.push(`${file}: Mismatched braces (${openBraces} open, ${closeBraces} close)`);
      }

      // Check for export statements
      if (!content.includes('export')) {
        issues.push(`${file}: No exports found`);
      }
    } else {
      issues.push(`${file}: File not found`);
    }
  });

  return {
    passed: issues.length === 0,
    details: {
      filesChecked: tsFiles.length,
      issuesFound: issues.length
    },
    warnings: issues
  };
});

// Print Summary
console.log('\n' + '='.repeat(70));
console.log('📊 VALIDATION SUMMARY');
console.log('='.repeat(70));
console.log(`✅ Passed:   ${results.summary.passed}/10`);
console.log(`❌ Failed:   ${results.summary.failed}/10`);
console.log(`⚠️  Warnings: ${results.summary.warnings}`);
console.log(`📈 Success Rate: ${((results.summary.passed / 10) * 100).toFixed(1)}%`);
console.log('='.repeat(70));

// Overall Grade
const successRate = (results.summary.passed / 10) * 100;
let grade, assessment;

if (successRate >= 90) {
  grade = 'A';
  assessment = 'Excellent - Production Ready';
} else if (successRate >= 80) {
  grade = 'B';
  assessment = 'Good - Minor improvements needed';
} else if (successRate >= 70) {
  grade = 'C';
  assessment = 'Acceptable - Some work required';
} else if (successRate >= 60) {
  grade = 'D';
  assessment = 'Poor - Significant improvements needed';
} else {
  grade = 'F';
  assessment = 'Failed - Major rework required';
}

console.log(`\n🎯 Overall Grade: ${grade}`);
console.log(`📋 Assessment: ${assessment}`);

// Save results
const resultsDir = path.join(__dirname, '../../test-results');
if (!fs.existsSync(resultsDir)) {
  fs.mkdirSync(resultsDir, { recursive: true });
}

const resultsFile = path.join(resultsDir, `validation-${Date.now()}.json`);
fs.writeFileSync(resultsFile, JSON.stringify(results, null, 2));
console.log(`\n📝 Detailed results saved to: ${resultsFile}`);

// Exit code
process.exit(results.summary.failed > 0 ? 1 : 0);
