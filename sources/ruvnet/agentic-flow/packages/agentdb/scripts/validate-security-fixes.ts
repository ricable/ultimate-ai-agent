#!/usr/bin/env node
/**
 * Security Validation Script
 *
 * Validates that all SQL injection vulnerabilities have been fixed.
 * Tests each fix with malicious inputs to ensure protection is working.
 *
 * Usage:
 *   npm run validate-security
 *   or
 *   ts-node scripts/validate-security-fixes.ts
 */

import {
  validateTableName,
  validateColumnName,
  validatePragmaCommand,
  validateSessionId,
  validateId,
  validateTimestamp,
  buildSafeWhereClause,
  buildSafeSetClause,
  ValidationError,
} from '../src/security/input-validation';

// ANSI color codes
const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const BLUE = '\x1b[34m';
const RESET = '\x1b[0m';

interface TestResult {
  name: string;
  passed: boolean;
  details?: string;
}

const results: TestResult[] = [];

function logTest(name: string, passed: boolean, details?: string) {
  results.push({ name, passed, details });
  const status = passed ? `${GREEN}✓ PASS${RESET}` : `${RED}✗ FAIL${RESET}`;
  console.log(`  ${status} ${name}`);
  if (details) {
    console.log(`         ${details}`);
  }
}

function testSection(title: string) {
  console.log(`\n${BLUE}${title}${RESET}`);
  console.log('='.repeat(60));
}

// ============================================================================
// Test 1: Table Name Injection Prevention
// ============================================================================
testSection('Test 1: Table Name Injection Prevention');

// SQL injection payloads for table names
const tableInjectionPayloads = [
  "episodes; DROP TABLE users--",
  "episodes' OR '1'='1",
  "episodes UNION SELECT * FROM users",
  "episodes'; DELETE FROM episodes--",
  "../../../etc/passwd",
  "episodes`; DELETE FROM skills;--",
  "episodes\x00malicious",
  "users", // Not in whitelist
  "sqlite_master", // System table
];

tableInjectionPayloads.forEach(payload => {
  try {
    validateTableName(payload);
    logTest(`Reject table injection: "${payload}"`, false, 'Payload was accepted!');
  } catch (error) {
    if (error instanceof ValidationError) {
      logTest(`Reject table injection: "${payload.substring(0, 30)}..."`, true);
    } else {
      logTest(`Reject table injection: "${payload}"`, false, `Wrong error type: ${error}`);
    }
  }
});

// Valid table names should pass
['episodes', 'skills', 'causal_edges', 'EPISODES'].forEach(table => {
  try {
    const result = validateTableName(table);
    logTest(`Accept valid table: "${table}"`, result === table.toLowerCase());
  } catch (error) {
    logTest(`Accept valid table: "${table}"`, false, `Rejected valid table: ${error}`);
  }
});

// ============================================================================
// Test 2: PRAGMA Command Injection Prevention
// ============================================================================
testSection('Test 2: PRAGMA Command Injection Prevention');

// SQL injection payloads for PRAGMA
const pragmaInjectionPayloads = [
  "journal_mode; DROP TABLE episodes--",
  "user_version; DELETE FROM skills--",
  "database_list; ATTACH 'malicious.db' AS evil",
  "compile_options'; DROP TABLE users--",
  "table_info(episodes)", // Information disclosure
  "database_list", // Information disclosure
];

pragmaInjectionPayloads.forEach(payload => {
  try {
    validatePragmaCommand(payload);
    logTest(`Reject PRAGMA injection: "${payload}"`, false, 'Payload was accepted!');
  } catch (error) {
    if (error instanceof ValidationError) {
      logTest(`Reject PRAGMA injection: "${payload.substring(0, 30)}..."`, true);
    } else {
      logTest(`Reject PRAGMA injection: "${payload}"`, false, `Wrong error type: ${error}`);
    }
  }
});

// Valid PRAGMA commands should pass
['journal_mode', 'cache_size', 'page_size', 'foreign_keys'].forEach(pragma => {
  try {
    const result = validatePragmaCommand(pragma);
    logTest(`Accept valid PRAGMA: "${pragma}"`, result.includes(pragma));
  } catch (error) {
    logTest(`Accept valid PRAGMA: "${pragma}"`, false, `Rejected valid PRAGMA: ${error}`);
  }
});

// ============================================================================
// Test 3: Session ID Injection Prevention
// ============================================================================
testSection('Test 3: Session ID Injection Prevention');

// SQL injection payloads for session IDs
const sessionIdInjectionPayloads = [
  "session' OR '1'='1",
  "session; DROP TABLE episodes--",
  "session' UNION SELECT * FROM users--",
  "session\x00malicious",
  "session OR 1=1",
  "session<script>alert(1)</script>",
  "session'; DELETE FROM episodes WHERE '1'='1",
];

sessionIdInjectionPayloads.forEach(payload => {
  try {
    validateSessionId(payload);
    logTest(`Reject session ID injection: "${payload}"`, false, 'Payload was accepted!');
  } catch (error) {
    if (error instanceof ValidationError) {
      logTest(`Reject session ID injection: "${payload.substring(0, 30)}..."`, true);
    } else {
      logTest(`Reject session ID injection: "${payload}"`, false, `Wrong error type: ${error}`);
    }
  }
});

// Valid session IDs should pass
['session-123', 'user_session_abc', 'a1b2c3d4e5'].forEach(sessionId => {
  try {
    const result = validateSessionId(sessionId);
    logTest(`Accept valid session ID: "${sessionId}"`, result === sessionId);
  } catch (error) {
    logTest(`Accept valid session ID: "${sessionId}"`, false, `Rejected valid ID: ${error}`);
  }
});

// ============================================================================
// Test 4: Numeric ID Injection Prevention
// ============================================================================
testSection('Test 4: Numeric ID Injection Prevention');

// SQL injection payloads for IDs
const idInjectionPayloads = [
  "1 OR 1=1",
  "1; DROP TABLE episodes--",
  "1 UNION SELECT * FROM users",
  "1' OR '1'='1",
  -1, // Negative ID
  3.14, // Non-integer
  NaN,
  Infinity,
];

idInjectionPayloads.forEach(payload => {
  try {
    validateId(payload);
    logTest(`Reject ID injection: "${payload}"`, false, 'Payload was accepted!');
  } catch (error) {
    if (error instanceof ValidationError) {
      logTest(`Reject ID injection: "${payload}"`, true);
    } else {
      logTest(`Reject ID injection: "${payload}"`, false, `Wrong error type: ${error}`);
    }
  }
});

// Valid IDs should pass
[0, 1, 42, 999, '123'].forEach(id => {
  try {
    const result = validateId(id);
    logTest(`Accept valid ID: ${id}`, typeof result === 'number' && result >= 0);
  } catch (error) {
    logTest(`Accept valid ID: ${id}`, false, `Rejected valid ID: ${error}`);
  }
});

// ============================================================================
// Test 5: WHERE Clause Building Security
// ============================================================================
testSection('Test 5: WHERE Clause Building Security');

// Test safe WHERE clause construction
try {
  const result = buildSafeWhereClause('episodes', {
    session_id: 'test',
    success: 1,
  });

  const passed = result.clause === 'session_id = ? AND success = ?' &&
                 result.values.length === 2 &&
                 result.values[0] === 'test' &&
                 result.values[1] === 1;

  logTest('Build safe WHERE clause with valid inputs', passed);
} catch (error) {
  logTest('Build safe WHERE clause with valid inputs', false, `${error}`);
}

// Test injection prevention in WHERE clause
try {
  buildSafeWhereClause('episodes', {
    "id' OR '1'='1": 1,
  });
  logTest('Reject WHERE clause column injection', false, 'Malicious column accepted!');
} catch (error) {
  if (error instanceof ValidationError) {
    logTest('Reject WHERE clause column injection', true);
  } else {
    logTest('Reject WHERE clause column injection', false, `Wrong error: ${error}`);
  }
}

// ============================================================================
// Test 6: SET Clause Building Security
// ============================================================================
testSection('Test 6: SET Clause Building Security');

// Test safe SET clause construction
try {
  const result = buildSafeSetClause('episodes', {
    reward: 0.95,
    success: 1,
  });

  const passed = result.clause === 'reward = ?, success = ?' &&
                 result.values.length === 2 &&
                 result.values[0] === 0.95 &&
                 result.values[1] === 1;

  logTest('Build safe SET clause with valid inputs', passed);
} catch (error) {
  logTest('Build safe SET clause with valid inputs', false, `${error}`);
}

// Test injection prevention in SET clause
try {
  buildSafeSetClause('episodes', {
    "reward = 1; DROP TABLE users--": 0.5,
  });
  logTest('Reject SET clause column injection', false, 'Malicious column accepted!');
} catch (error) {
  if (error instanceof ValidationError) {
    logTest('Reject SET clause column injection', true);
  } else {
    logTest('Reject SET clause column injection', false, `Wrong error: ${error}`);
  }
}

// ============================================================================
// Real-World Attack Scenarios
// ============================================================================
testSection('Real-World Attack Scenarios');

// Bobby Tables attack
try {
  validateSessionId("Robert'); DROP TABLE episodes;--");
  logTest('Prevent Bobby Tables attack', false, 'Attack payload accepted!');
} catch (error) {
  if (error instanceof ValidationError) {
    logTest('Prevent Bobby Tables attack', true);
  } else {
    logTest('Prevent Bobby Tables attack', false, `Wrong error: ${error}`);
  }
}

// UNION-based data exfiltration
try {
  validateColumnName('episodes', "id UNION SELECT password FROM users");
  logTest('Prevent UNION-based exfiltration', false, 'Attack payload accepted!');
} catch (error) {
  if (error instanceof ValidationError) {
    logTest('Prevent UNION-based exfiltration', true);
  } else {
    logTest('Prevent UNION-based exfiltration', false, `Wrong error: ${error}`);
  }
}

// Stacked queries
try {
  validateSessionId("'; DELETE FROM episodes; --");
  logTest('Prevent stacked queries', false, 'Attack payload accepted!');
} catch (error) {
  if (error instanceof ValidationError) {
    logTest('Prevent stacked queries', true);
  } else {
    logTest('Prevent stacked queries', false, `Wrong error: ${error}`);
  }
}

// Time-based blind SQL injection
try {
  buildSafeWhereClause('episodes', {
    "id AND SLEEP(5)": 1,
  });
  logTest('Prevent time-based blind injection', false, 'Attack payload accepted!');
} catch (error) {
  if (error instanceof ValidationError) {
    logTest('Prevent time-based blind injection', true);
  } else {
    logTest('Prevent time-based blind injection', false, `Wrong error: ${error}`);
  }
}

// ============================================================================
// Summary
// ============================================================================
console.log('\n' + '='.repeat(60));
console.log(`${BLUE}VALIDATION SUMMARY${RESET}`);
console.log('='.repeat(60));

const totalTests = results.length;
const passedTests = results.filter(r => r.passed).length;
const failedTests = totalTests - passedTests;

console.log(`Total Tests:  ${totalTests}`);
console.log(`${GREEN}Passed:       ${passedTests}${RESET}`);
if (failedTests > 0) {
  console.log(`${RED}Failed:       ${failedTests}${RESET}`);
}

const passRate = ((passedTests / totalTests) * 100).toFixed(1);
console.log(`Pass Rate:    ${passRate}%`);

// Show failed tests if any
if (failedTests > 0) {
  console.log(`\n${RED}FAILED TESTS:${RESET}`);
  results.filter(r => !r.passed).forEach(result => {
    console.log(`  ${RED}✗${RESET} ${result.name}`);
    if (result.details) {
      console.log(`    ${result.details}`);
    }
  });
}

// Exit with appropriate code
if (failedTests > 0) {
  console.log(`\n${RED}❌ Security validation FAILED${RESET}`);
  process.exit(1);
} else {
  console.log(`\n${GREEN}✅ All security validations PASSED${RESET}`);
  console.log(`${GREEN}🔒 SQL injection vulnerabilities are fixed!${RESET}`);
  process.exit(0);
}
