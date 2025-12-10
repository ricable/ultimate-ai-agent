/**
 * Agentic-Jujutsu Billing System Examples
 * Demonstrates common usage patterns for the TypeScript billing system
 */

import {
  createBillingSystem,
  SubscriptionTier,
  BillingCycle,
  UsageMetric,
  CouponType
} from '../agentic-flow/src/billing/index.js';

async function main() {
  console.log('🚀 Agentic-Jujutsu Billing System Examples\n');

  // ============================================
  // Example 1: Basic Subscription Creation
  // ============================================
  console.log('Example 1: Creating a subscription');
  console.log('='.repeat(50));

  const billing = createBillingSystem({
    currency: 'USD',
    enableMetering: true,
    enableCoupons: true
  });

  const { subscription, payment } = await billing.subscribe({
    userId: 'demo_user_1',
    tier: SubscriptionTier.Pro,
    billingCycle: BillingCycle.Monthly,
    paymentMethodId: 'pm_demo_card_123'
  });

  console.log(`✅ Subscription created: ${subscription.id}`);
  console.log(`   Tier: ${subscription.tier}`);
  console.log(`   Price: $${subscription.price}/${subscription.billingCycle}`);
  console.log(`   Status: ${subscription.status}`);
  console.log(`   Payment: ${payment.status}\n`);

  // ============================================
  // Example 2: Usage Tracking
  // ============================================
  console.log('Example 2: Recording and tracking usage');
  console.log('='.repeat(50));

  // Simulate agent usage over time
  console.log('Recording agent usage...');
  for (let i = 0; i < 5; i++) {
    await billing.recordUsage({
      subscriptionId: subscription.id,
      userId: 'demo_user_1',
      metric: UsageMetric.AgentHours,
      amount: 100,
      unit: 'hours'
    });
    console.log(`  ✓ Recorded ${100 * (i + 1)} agent hours`);
  }

  // Get usage summary
  const summary = await billing.getUsageSummary(subscription.id);
  const agentHours = summary.metrics.get(UsageMetric.AgentHours) || 0;
  const percentUsed = summary.percentUsed.get(UsageMetric.AgentHours) || 0;

  console.log(`\n📊 Usage Summary:`);
  console.log(`   Agent hours: ${agentHours}/2000 (${percentUsed.toFixed(1)}%)`);
  console.log(`   Status: ${percentUsed < 80 ? '✅ Under quota' : '⚠️  Approaching limit'}\n`);

  // ============================================
  // Example 3: Quota Checking
  // ============================================
  console.log('Example 3: Quota checking before operations');
  console.log('='.repeat(50));

  const quotaOk = await billing.checkQuota(subscription.id, UsageMetric.AgentHours);

  if (quotaOk) {
    console.log('✅ Quota check passed - operation allowed');
    console.log('   Proceeding with agent execution...\n');
  } else {
    console.log('❌ Quota exceeded - operation blocked');
    console.log('   Please upgrade your plan\n');
  }

  // ============================================
  // Example 4: Subscription Upgrade
  // ============================================
  console.log('Example 4: Upgrading subscription tier');
  console.log('='.repeat(50));

  console.log(`Current tier: ${subscription.tier} ($${subscription.price}/mo)`);

  const upgraded = await billing.upgrade(subscription.id, SubscriptionTier.Enterprise);

  console.log(`✅ Upgraded to: ${upgraded.tier} ($${upgraded.price}/mo)`);
  console.log(`   New limits:`);
  console.log(`   - Agent hours: ${upgraded.limits.maxAgentHours}/month`);
  console.log(`   - Swarm size: up to ${upgraded.limits.maxSwarmSize}`);
  console.log(`   - Storage: ${upgraded.limits.maxStorageGB}GB\n`);

  // ============================================
  // Example 5: Coupon System
  // ============================================
  console.log('Example 5: Creating and using coupons');
  console.log('='.repeat(50));

  // Create a 25% off coupon
  const coupon = await billing.coupons.createCoupon({
    code: 'DEMO25',
    type: CouponType.Percentage,
    value: 25,
    description: '25% off for demo',
    maxRedemptions: 100
  });

  console.log(`✅ Coupon created: ${coupon.code}`);
  console.log(`   Type: ${coupon.type}`);
  console.log(`   Value: ${coupon.value}%`);

  // Validate coupon
  const validation = await billing.coupons.validateCoupon(
    'DEMO25',
    SubscriptionTier.Pro,
    99
  );

  if (validation.valid) {
    console.log(`   Valid! Discount: $${validation.discountAmount.toFixed(2)}`);
    console.log(`   Final price: $${validation.finalAmount.toFixed(2)}\n`);
  }

  // ============================================
  // Example 6: Event Listeners
  // ============================================
  console.log('Example 6: Setting up event listeners');
  console.log('='.repeat(50));

  billing.on('usage.recorded', (data) => {
    console.log(`📊 Usage recorded: ${data.amount} ${data.unit} of ${data.metric}`);
  });

  billing.on('quota.warning', (data) => {
    console.log(`⚠️  Warning: ${data.percentUsed.toFixed(1)}% of ${data.metric} quota used`);
  });

  billing.on('quota.exceeded', (data) => {
    console.log(`🚫 Quota exceeded for ${data.metric}`);
    console.log(`   Current: ${data.current}, Limit: ${data.limit}`);
  });

  billing.on('subscription.upgraded', (data) => {
    console.log(`🎉 Subscription upgraded: ${data.oldTier} → ${data.newTier}`);
  });

  console.log('✅ Event listeners configured\n');

  // Record some usage to trigger events
  console.log('Recording usage to demonstrate events:');
  await billing.recordUsage({
    subscriptionId: subscription.id,
    userId: 'demo_user_1',
    metric: UsageMetric.APIRequests,
    amount: 50000,
    unit: 'requests'
  });

  // ============================================
  // Example 7: Pricing Comparison
  // ============================================
  console.log('\nExample 7: Comparing pricing tiers');
  console.log('='.repeat(50));

  const comparison = billing.pricing.compareFeatures(
    SubscriptionTier.Starter,
    SubscriptionTier.Pro
  );

  console.log('Starter → Pro upgrade benefits:');
  console.log('   Multipliers:');
  Object.entries(comparison.multipliers).forEach(([key, mult]) => {
    console.log(`   - ${key}: ${mult}x more`);
  });

  console.log('\n   New features:');
  comparison.upgrades.slice(0, 3).forEach(feature => {
    console.log(`   + ${feature}`);
  });

  // ============================================
  // Example 8: Tier Recommendation
  // ============================================
  console.log('\nExample 8: Getting tier recommendations');
  console.log('='.repeat(50));

  const scenarios = [
    { maxAgentHours: 150, maxStorageGB: 5 },
    { maxAgentHours: 750, maxStorageGB: 20 },
    { maxAgentHours: 3000, maxStorageGB: 150 }
  ];

  scenarios.forEach(usage => {
    const recommended = billing.pricing.getRecommendedTier(usage);
    console.log(`Usage: ${usage.maxAgentHours} hours, ${usage.maxStorageGB}GB`);
    console.log(`   → Recommended: ${recommended}\n`);
  });

  // ============================================
  // Example 9: Multiple Subscriptions
  // ============================================
  console.log('Example 9: Managing multiple subscriptions');
  console.log('='.repeat(50));

  // Create additional subscriptions
  const starterSub = await billing.subscriptions.createSubscription({
    userId: 'demo_user_2',
    tier: SubscriptionTier.Starter,
    billingCycle: BillingCycle.Monthly,
    paymentMethodId: 'pm_demo_card_456'
  });

  const freeSub = await billing.subscriptions.createSubscription({
    userId: 'demo_user_3',
    tier: SubscriptionTier.Free,
    billingCycle: BillingCycle.Monthly,
    paymentMethodId: 'pm_demo_card_789'
  });

  console.log('Created multiple subscriptions:');
  console.log(`   1. ${subscription.tier} - $${subscription.price}/mo`);
  console.log(`   2. ${starterSub.tier} - $${starterSub.price}/mo`);
  console.log(`   3. ${freeSub.tier} - $${freeSub.price}/mo\n`);

  // ============================================
  // Example 10: Subscription Cancellation
  // ============================================
  console.log('Example 10: Canceling a subscription');
  console.log('='.repeat(50));

  const toCancel = await billing.subscriptions.createSubscription({
    userId: 'demo_user_cancel',
    tier: SubscriptionTier.Starter,
    billingCycle: BillingCycle.Monthly,
    paymentMethodId: 'pm_demo_cancel'
  });

  console.log(`Subscription created: ${toCancel.id}`);

  // Cancel at period end
  const canceled = await billing.cancel(toCancel.id, false);

  console.log(`✅ Cancellation scheduled`);
  console.log(`   Active until: ${canceled.currentPeriodEnd}`);
  console.log(`   Cancel at period end: ${canceled.cancelAtPeriodEnd}\n`);

  // ============================================
  // Summary
  // ============================================
  console.log('='.repeat(50));
  console.log('✅ All examples completed successfully!');
  console.log('\nKey Features Demonstrated:');
  console.log('   ✓ Subscription creation with payment');
  console.log('   ✓ Usage tracking and metering');
  console.log('   ✓ Quota checking and enforcement');
  console.log('   ✓ Subscription upgrades');
  console.log('   ✓ Coupon creation and validation');
  console.log('   ✓ Event-driven notifications');
  console.log('   ✓ Pricing comparisons');
  console.log('   ✓ Tier recommendations');
  console.log('   ✓ Multi-user management');
  console.log('   ✓ Subscription cancellation');
  console.log('\n💡 Ready to integrate into your application!');

  // Cleanup
  await billing.shutdown();
}

// Run examples
main().catch(error => {
  console.error('Error running examples:', error);
  process.exit(1);
});
