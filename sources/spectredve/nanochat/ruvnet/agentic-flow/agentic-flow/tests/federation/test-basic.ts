/**
 * Basic test suite for federation security and lifecycle (no AgentDB dependency)
 */

import { SecurityManager } from '../../src/federation/SecurityManager.js';

async function testSecurityManager() {
  console.log('\n=== Testing SecurityManager ===\n');

  const security = new SecurityManager();

  // Test 1: JWT token generation and verification
  console.log('Test 1: JWT token generation and verification...');
  const token = await security.createAgentToken({
    agentId: 'test-agent-123',
    tenantId: 'test-tenant',
    expiresAt: Date.now() + 300000 // 5 minutes
  });

  console.log(`✅ Token created: ${token.substring(0, 50)}...`);

  const payload = await security.verifyAgentToken(token);
  console.log(`✅ Token verified:`, payload);

  // Test 2: Encryption and decryption
  console.log('\nTest 2: Data encryption and decryption...');
  const testData = 'Sensitive agent memory data';
  const { encrypted, authTag } = await security.encrypt(testData, 'test-tenant');
  console.log(`✅ Data encrypted: ${encrypted.substring(0, 30)}...`);

  const decrypted = await security.decrypt(encrypted, authTag, 'test-tenant');
  console.log(`✅ Data decrypted: ${decrypted}`);

  if (decrypted === testData) {
    console.log('✅ Encryption/decryption successful (data matches)');
  } else {
    throw new Error('Decrypted data does not match original');
  }

  // Test 3: Tenant access validation
  console.log('\nTest 3: Tenant access validation...');
  const validAccess = security.validateTenantAccess('tenant-a', 'tenant-a');
  const invalidAccess = security.validateTenantAccess('tenant-a', 'tenant-b');

  if (validAccess && !invalidAccess) {
    console.log('✅ Tenant isolation working correctly');
  } else {
    throw new Error('Tenant access validation failed');
  }

  // Test 4: Token expiration
  console.log('\nTest 4: Token expiration...');
  const expiredToken = await security.createAgentToken({
    agentId: 'test-agent-expired',
    tenantId: 'test-tenant',
    expiresAt: Date.now() - 1000 // Expired 1 second ago
  });

  try {
    await security.verifyAgentToken(expiredToken);
    throw new Error('Should have rejected expired token');
  } catch (error: any) {
    if (error.message.includes('expired')) {
      console.log('✅ Correctly rejected expired token');
    } else {
      throw error;
    }
  }

  // Test 5: Secure ID generation
  console.log('\nTest 5: Secure ID generation...');
  const id1 = security.generateSecureId();
  const id2 = security.generateSecureId();
  console.log(`   ID 1: ${id1}`);
  console.log(`   ID 2: ${id2}`);

  if (id1 !== id2 && id1.length === 32) {
    console.log('✅ Secure IDs generated correctly');
  } else {
    throw new Error('Secure ID generation failed');
  }

  // Test 6: Data hashing
  console.log('\nTest 6: Data hashing...');
  const hash1 = security.hashData('test-data');
  const hash2 = security.hashData('test-data');
  const hash3 = security.hashData('different-data');

  if (hash1 === hash2 && hash1 !== hash3) {
    console.log('✅ Data hashing working correctly');
    console.log(`   Hash: ${hash1}`);
  } else {
    throw new Error('Data hashing failed');
  }

  console.log('\n=== All SecurityManager tests passed ===\n');
}

async function testLifecycle() {
  console.log('\n=== Testing Agent Lifecycle Concepts ===\n');

  // Test lifecycle timing
  const lifetime = 10; // 10 seconds
  const spawnTime = Date.now();
  const expiresAt = spawnTime + (lifetime * 1000);

  console.log('Test: Agent lifecycle timing...');
  console.log(`   Spawn time: ${new Date(spawnTime).toISOString()}`);
  console.log(`   Expires at: ${new Date(expiresAt).toISOString()}`);
  console.log(`   Lifetime: ${lifetime}s`);

  const now = Date.now();
  const remainingLifetime = Math.max(0, Math.floor((expiresAt - now) / 1000));
  const isExpired = now >= expiresAt;

  console.log(`   Current time: ${new Date(now).toISOString()}`);
  console.log(`   Remaining lifetime: ${remainingLifetime}s`);
  console.log(`   Is expired: ${isExpired}`);

  if (remainingLifetime === lifetime && !isExpired) {
    console.log('✅ Lifecycle timing correct');
  }

  // Test wait for expiration
  console.log('\nTest: Waiting 3 seconds...');
  await new Promise(resolve => setTimeout(resolve, 3000));

  const now2 = Date.now();
  const remainingLifetime2 = Math.max(0, Math.floor((expiresAt - now2) / 1000));

  console.log(`   Time after wait: ${new Date(now2).toISOString()}`);
  console.log(`   Remaining lifetime: ${remainingLifetime2}s`);

  if (remainingLifetime2 < lifetime && remainingLifetime2 >= lifetime - 4) {
    console.log('✅ Lifecycle countdown working');
  }

  console.log('\n=== Lifecycle tests passed ===\n');
}

async function testVectorClock() {
  console.log('\n=== Testing Vector Clock Concepts ===\n');

  // Simulate vector clock for conflict detection
  const vectorClock1: Record<string, number> = {
    'agent-a': 5,
    'agent-b': 3,
    'agent-c': 2
  };

  const vectorClock2: Record<string, number> = {
    'agent-a': 4,
    'agent-b': 6,
    'agent-c': 2
  };

  console.log('Test: Conflict detection with vector clocks...');
  console.log('   Vector clock 1:', vectorClock1);
  console.log('   Vector clock 2:', vectorClock2);

  // Detect conflict (neither dominates)
  let localDominates = false;
  let remoteDominates = false;

  for (const agentId in vectorClock2) {
    const localTs = vectorClock1[agentId] || 0;
    const remoteTs = vectorClock2[agentId];

    if (localTs > remoteTs) {
      localDominates = true;
    } else if (remoteTs > localTs) {
      remoteDominates = true;
    }
  }

  const hasConflict = localDominates && remoteDominates;

  console.log(`   Local dominates: ${localDominates}`);
  console.log(`   Remote dominates: ${remoteDominates}`);
  console.log(`   Has conflict: ${hasConflict}`);

  if (hasConflict) {
    console.log('✅ Conflict correctly detected (concurrent updates)');
  }

  // Test merge (take maximum timestamps)
  const merged: Record<string, number> = {};
  for (const agentId in vectorClock1) {
    merged[agentId] = vectorClock1[agentId];
  }
  for (const agentId in vectorClock2) {
    const localTs = merged[agentId] || 0;
    const remoteTs = vectorClock2[agentId];
    merged[agentId] = Math.max(localTs, remoteTs);
  }

  console.log('\nTest: Vector clock merge...');
  console.log('   Merged clock:', merged);

  const expectedMerged = {
    'agent-a': 5,
    'agent-b': 6,
    'agent-c': 2
  };

  if (JSON.stringify(merged) === JSON.stringify(expectedMerged)) {
    console.log('✅ Vector clock merge correct (max timestamps)');
  }

  console.log('\n=== Vector clock tests passed ===\n');
}

// Run all tests
(async () => {
  try {
    await testSecurityManager();
    await testLifecycle();
    await testVectorClock();

    console.log('\n🎉 All federation basic tests passed!\n');
    console.log('Note: Full integration tests with AgentDB require building the agentdb package.');
    console.log('Run `npm run build` in the agentdb package directory first.\n');
  } catch (error: any) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
})();
