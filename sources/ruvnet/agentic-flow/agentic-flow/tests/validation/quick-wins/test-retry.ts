#!/usr/bin/env tsx
// Test retry mechanism
import { withRetry } from '../../src/utils/retry.js';

console.log('🔄 Testing Retry Mechanism\n');

// Test 1: Successful operation
console.log('Test 1: Successful operation (should succeed immediately)');
let attempt1 = 0;
try {
  await withRetry(async () => {
    attempt1++;
    console.log(`  Attempt ${attempt1}`);
    return 'success';
  });
  console.log('✅ Passed - Succeeded on first attempt\n');
} catch (error) {
  console.error('❌ Failed - Should have succeeded');
  process.exit(1);
}

// Test 2: Retryable error
console.log('Test 2: Retryable error (should retry 3 times)');
let attempt2 = 0;
try {
  await withRetry(
    async () => {
      attempt2++;
      console.log(`  Attempt ${attempt2}`);
      if (attempt2 < 3) {
        const error: any = new Error('Server error');
        error.status = 500;
        throw error;
      }
      return 'success after retries';
    },
    { maxAttempts: 3, baseDelay: 100 }
  );
  console.log(`✅ Passed - Succeeded after ${attempt2} attempts\n`);
} catch (error) {
  console.error('❌ Failed - Should have succeeded after retries');
  process.exit(1);
}

// Test 3: Non-retryable error
console.log('Test 3: Non-retryable error (should fail immediately)');
let attempt3 = 0;
try {
  await withRetry(
    async () => {
      attempt3++;
      console.log(`  Attempt ${attempt3}`);
      const error: any = new Error('Bad request');
      error.status = 400;
      throw error;
    },
    { maxAttempts: 3, baseDelay: 100 }
  );
  console.error('❌ Failed - Should have thrown an error');
  process.exit(1);
} catch (error) {
  if (attempt3 === 1) {
    console.log('✅ Passed - Failed immediately without retries\n');
  } else {
    console.error('❌ Failed - Should not have retried');
    process.exit(1);
  }
}

// Test 4: Max retries exceeded
console.log('Test 4: Max retries exceeded (should fail after 3 attempts)');
let attempt4 = 0;
try {
  await withRetry(
    async () => {
      attempt4++;
      console.log(`  Attempt ${attempt4}`);
      const error: any = new Error('Always fails');
      error.status = 500;
      throw error;
    },
    { maxAttempts: 3, baseDelay: 100 }
  );
  console.error('❌ Failed - Should have thrown an error');
  process.exit(1);
} catch (error) {
  if (attempt4 === 3) {
    console.log('✅ Passed - Failed after max retries\n');
  } else {
    console.error(`❌ Failed - Should have retried exactly 3 times, got ${attempt4}`);
    process.exit(1);
  }
}

console.log('═══════════════════════════════════════════════════════');
console.log('✅ All retry tests passed!');
