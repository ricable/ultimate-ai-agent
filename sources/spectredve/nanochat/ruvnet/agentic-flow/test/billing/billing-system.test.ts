/**
 * Comprehensive Test Suite for TypeScript Billing System
 * Tests all components: Pricing, Metering, Subscriptions, Coupons, Payments
 */

import {
  BillingSystem,
  SubscriptionTier,
  BillingCycle,
  UsageMetric,
  CouponType,
  PaymentProvider
} from '../../agentic-flow/src/billing/index.js';

interface TestResult {
  category: string;
  testName: string;
  passed: boolean;
  details?: string;
  error?: string;
}

class BillingSystemTest {
  private results: TestResult[] = [];
  private billing!: BillingSystem;

  async runAllTests(): Promise<void> {
    console.log('🧪 Starting Comprehensive Billing System Tests\n');
    console.log('=' .repeat(60));

    // Initialize billing system
    this.billing = new BillingSystem({
      currency: 'USD',
      enableMetering: true,
      enableCoupons: true,
      storageBackend: 'memory'
    });

    // Run test suites
    await this.testPricingSystem();
    await this.testMeteringEngine();
    await this.testSubscriptions();
    await this.testCoupons();
    await this.testQuotaManagement();
    await this.testIntegration();

    // Shutdown
    await this.billing.shutdown();

    // Report results
    this.reportResults();
  }

  private async testPricingSystem(): Promise<void> {
    console.log('\n📊 Testing Pricing System...');

    try {
      // Test 1: List all tiers
      const tiers = this.billing.pricing.getAllTiers();
      this.addResult({
        category: 'Pricing',
        testName: 'List All Tiers',
        passed: tiers.length === 5,
        details: `Found ${tiers.length} tiers`
      });

      // Test 2: Get specific tier
      const proTier = this.billing.pricing.getTier(SubscriptionTier.Pro);
      this.addResult({
        category: 'Pricing',
        testName: 'Get Pro Tier',
        passed: proTier !== undefined && proTier.monthlyPrice === 99,
        details: `Pro tier: $${proTier?.monthlyPrice}/mo`
      });

      // Test 3: Calculate yearly price
      const yearlyPrice = this.billing.pricing.calculatePrice(SubscriptionTier.Pro, 'yearly');
      this.addResult({
        category: 'Pricing',
        testName: 'Calculate Yearly Price',
        passed: yearlyPrice === 990,
        details: `Yearly price: $${yearlyPrice} (2 months free)`
      });

      // Test 4: Annual savings
      const savings = this.billing.pricing.getAnnualSavings(SubscriptionTier.Pro);
      this.addResult({
        category: 'Pricing',
        testName: 'Calculate Annual Savings',
        passed: savings === 198,
        details: `Annual savings: $${savings}`
      });

      // Test 5: Compare tiers
      const comparison = this.billing.pricing.compareFeatures(
        SubscriptionTier.Starter,
        SubscriptionTier.Pro
      );
      this.addResult({
        category: 'Pricing',
        testName: 'Compare Tiers',
        passed: comparison.multipliers.agentHours === 4,
        details: `Pro has ${comparison.multipliers.agentHours}x agent hours vs Starter`
      });

      // Test 6: Recommend tier
      const recommended = this.billing.pricing.getRecommendedTier({
        maxAgentHours: 1500
      });
      this.addResult({
        category: 'Pricing',
        testName: 'Recommend Tier',
        passed: recommended === SubscriptionTier.Pro,
        details: `Recommended tier for 1500 hours: ${recommended}`
      });
    } catch (error) {
      this.addResult({
        category: 'Pricing',
        testName: 'Pricing System',
        passed: false,
        error: String(error)
      });
    }
  }

  private async testMeteringEngine(): Promise<void> {
    console.log('\n📈 Testing Metering Engine...');

    try {
      // Create test subscription
      const sub = await this.billing.subscriptions.createSubscription({
        userId: 'test-user-1',
        tier: SubscriptionTier.Pro,
        billingCycle: BillingCycle.Monthly,
        paymentMethodId: 'pm_test'
      });

      // Test 1: Record usage
      await this.billing.recordUsage({
        subscriptionId: sub.id,
        userId: sub.userId,
        metric: UsageMetric.AgentHours,
        amount: 100,
        unit: 'hours'
      });

      this.addResult({
        category: 'Metering',
        testName: 'Record Usage',
        passed: true,
        details: 'Recorded 100 agent hours'
      });

      // Test 2: Get usage summary
      const summary = await this.billing.getUsageSummary(sub.id);
      const agentHoursUsed = summary.metrics.get(UsageMetric.AgentHours) || 0;

      this.addResult({
        category: 'Metering',
        testName: 'Usage Summary',
        passed: agentHoursUsed === 100,
        details: `Agent hours used: ${agentHoursUsed}/2000`
      });

      // Test 3: Percentage calculation
      const percentUsed = summary.percentUsed.get(UsageMetric.AgentHours) || 0;
      this.addResult({
        category: 'Metering',
        testName: 'Percentage Calculation',
        passed: percentUsed === 5, // 100/2000 = 5%
        details: `${percentUsed.toFixed(1)}% used`
      });

      // Test 4: Check quota (should pass)
      const quotaOk = await this.billing.checkQuota(sub.id, UsageMetric.AgentHours);
      this.addResult({
        category: 'Metering',
        testName: 'Quota Check (Pass)',
        passed: quotaOk === true,
        details: 'Quota check passed (under limit)'
      });

      // Test 5: Record heavy usage
      await this.billing.recordUsage({
        subscriptionId: sub.id,
        userId: sub.userId,
        metric: UsageMetric.AgentHours,
        amount: 1950,
        unit: 'hours'
      });

      // Test 6: Check quota (should warn/fail)
      const summary2 = await this.billing.getUsageSummary(sub.id);
      const totalUsed = summary2.metrics.get(UsageMetric.AgentHours) || 0;

      this.addResult({
        category: 'Metering',
        testName: 'Heavy Usage Tracking',
        passed: totalUsed === 2050,
        details: `Total usage: ${totalUsed} hours (2.5% over limit)`
      });

      // Test 7: Overage detection
      const hasOverage = summary2.overages.has(UsageMetric.AgentHours);
      this.addResult({
        category: 'Metering',
        testName: 'Overage Detection',
        passed: hasOverage,
        details: `Overage amount: ${summary2.overages.get(UsageMetric.AgentHours)} hours`
      });

      // Test 8: Overage cost calculation
      const overageCost = summary2.estimatedCost;
      this.addResult({
        category: 'Metering',
        testName: 'Overage Cost Calculation',
        passed: overageCost > 0,
        details: `Estimated overage cost: $${overageCost.toFixed(2)}`
      });
    } catch (error) {
      this.addResult({
        category: 'Metering',
        testName: 'Metering Engine',
        passed: false,
        error: String(error)
      });
    }
  }

  private async testSubscriptions(): Promise<void> {
    console.log('\n💳 Testing Subscription Management...');

    try {
      // Test 1: Create subscription
      const sub = await this.billing.subscriptions.createSubscription({
        userId: 'test-user-2',
        tier: SubscriptionTier.Starter,
        billingCycle: BillingCycle.Monthly,
        paymentMethodId: 'pm_test_2'
      });

      this.addResult({
        category: 'Subscriptions',
        testName: 'Create Subscription',
        passed: sub.tier === SubscriptionTier.Starter && sub.price === 29,
        details: `Created Starter subscription: $${sub.price}/month`
      });

      // Test 2: Upgrade subscription
      const upgraded = await this.billing.subscriptions.upgradeSubscription(
        sub.id,
        SubscriptionTier.Pro
      );

      this.addResult({
        category: 'Subscriptions',
        testName: 'Upgrade Subscription',
        passed: upgraded.tier === SubscriptionTier.Pro && upgraded.price === 99,
        details: `Upgraded to Pro: $${upgraded.price}/month`
      });

      // Test 3: Downgrade subscription
      const downgraded = await this.billing.subscriptions.downgradeSubscription(
        sub.id,
        SubscriptionTier.Starter
      );

      this.addResult({
        category: 'Subscriptions',
        testName: 'Downgrade Subscription',
        passed: downgraded.metadata?.pendingDowngrade === SubscriptionTier.Starter,
        details: 'Downgrade scheduled for end of period'
      });

      // Test 4: List subscriptions
      const subs = await this.billing.subscriptions.listSubscriptions('test-user-2');
      this.addResult({
        category: 'Subscriptions',
        testName: 'List Subscriptions',
        passed: subs.length >= 1,
        details: `Found ${subs.length} subscription(s)`
      });

      // Test 5: Cancel subscription
      const canceled = await this.billing.subscriptions.cancelSubscription(sub.id, false);
      this.addResult({
        category: 'Subscriptions',
        testName: 'Cancel Subscription',
        passed: canceled.cancelAtPeriodEnd === true,
        details: 'Cancellation scheduled for end of period'
      });

      // Test 6: Feature access check
      const hasAccess = await this.billing.subscriptions.hasAccess(
        sub.id,
        'ReasoningBank learning'
      );
      this.addResult({
        category: 'Subscriptions',
        testName: 'Feature Access Check',
        passed: hasAccess === true,
        details: 'Pro tier has ReasoningBank access'
      });
    } catch (error) {
      this.addResult({
        category: 'Subscriptions',
        testName: 'Subscription Management',
        passed: false,
        error: String(error)
      });
    }
  }

  private async testCoupons(): Promise<void> {
    console.log('\n🎟️  Testing Coupon System...');

    try {
      // Test 1: Create percentage coupon
      const percentCoupon = await this.billing.coupons.createCoupon({
        code: 'TEST20',
        type: CouponType.Percentage,
        value: 20,
        description: '20% off',
        maxRedemptions: 100
      });

      this.addResult({
        category: 'Coupons',
        testName: 'Create Percentage Coupon',
        passed: percentCoupon.code === 'TEST20' && percentCoupon.value === 20,
        details: '20% discount coupon created'
      });

      // Test 2: Validate coupon
      const validation = await this.billing.coupons.validateCoupon(
        'TEST20',
        SubscriptionTier.Pro,
        99
      );

      this.addResult({
        category: 'Coupons',
        testName: 'Validate Coupon',
        passed: validation.valid === true && validation.discountAmount === 19.8,
        details: `Discount: $${validation.discountAmount}, Final: $${validation.finalAmount}`
      });

      // Test 3: Create fixed coupon
      const fixedCoupon = await this.billing.coupons.createCoupon({
        code: 'FIXED50',
        type: CouponType.Fixed,
        value: 50,
        currency: 'USD',
        description: '$50 off'
      });

      this.addResult({
        category: 'Coupons',
        testName: 'Create Fixed Coupon',
        passed: fixedCoupon.type === CouponType.Fixed && fixedCoupon.value === 50,
        details: '$50 fixed discount created'
      });

      // Test 4: Apply coupon
      const applied = await this.billing.coupons.applyCoupon('TEST20');
      this.addResult({
        category: 'Coupons',
        testName: 'Apply Coupon',
        passed: applied.timesRedeemed === 1,
        details: `Coupon redeemed ${applied.timesRedeemed} time(s)`
      });

      // Test 5: List coupons
      const coupons = await this.billing.coupons.listCoupons(true);
      this.addResult({
        category: 'Coupons',
        testName: 'List Active Coupons',
        passed: coupons.length >= 2,
        details: `Found ${coupons.length} active coupon(s)`
      });

      // Test 6: Expired coupon
      const expiredCoupon = await this.billing.coupons.createCoupon({
        code: 'EXPIRED',
        type: CouponType.Percentage,
        value: 10,
        validFrom: new Date('2020-01-01'),
        validUntil: new Date('2020-12-31')
      });

      const expiredValidation = await this.billing.coupons.validateCoupon(
        'EXPIRED',
        SubscriptionTier.Pro,
        99
      );

      this.addResult({
        category: 'Coupons',
        testName: 'Expired Coupon Validation',
        passed: expiredValidation.valid === false,
        details: `Error: ${expiredValidation.error}`
      });
    } catch (error) {
      this.addResult({
        category: 'Coupons',
        testName: 'Coupon System',
        passed: false,
        error: String(error)
      });
    }
  }

  private async testQuotaManagement(): Promise<void> {
    console.log('\n🚦 Testing Quota Management...');

    try {
      const sub = await this.billing.subscriptions.createSubscription({
        userId: 'test-user-quota',
        tier: SubscriptionTier.Free,
        billingCycle: BillingCycle.Monthly,
        paymentMethodId: 'pm_test_quota'
      });

      // Test 1: Initial quota check
      const initialCheck = await this.billing.checkQuota(sub.id, UsageMetric.AgentHours);
      this.addResult({
        category: 'Quota',
        testName: 'Initial Quota Check',
        passed: initialCheck === true,
        details: 'Initial quota check passed'
      });

      // Test 2: Use 50% of quota
      await this.billing.recordUsage({
        subscriptionId: sub.id,
        userId: sub.userId,
        metric: UsageMetric.AgentHours,
        amount: 50,
        unit: 'hours'
      });

      const midCheck = await this.billing.checkQuota(sub.id, UsageMetric.AgentHours);
      this.addResult({
        category: 'Quota',
        testName: 'Mid-Range Quota Check',
        passed: midCheck === true,
        details: '50% quota used - still allowed'
      });

      // Test 3: Approach soft limit (80%)
      await this.billing.recordUsage({
        subscriptionId: sub.id,
        userId: sub.userId,
        metric: UsageMetric.AgentHours,
        amount: 35,
        unit: 'hours'
      });

      const summary = await this.billing.getUsageSummary(sub.id);
      const percentUsed = summary.percentUsed.get(UsageMetric.AgentHours) || 0;

      this.addResult({
        category: 'Quota',
        testName: 'Soft Limit Warning',
        passed: percentUsed >= 80 && percentUsed < 100,
        details: `${percentUsed.toFixed(1)}% used - warning triggered`
      });

      // Test 4: Exceed hard limit (100%)
      await this.billing.recordUsage({
        subscriptionId: sub.id,
        userId: sub.userId,
        metric: UsageMetric.AgentHours,
        amount: 20,
        unit: 'hours'
      });

      const hardCheck = await this.billing.checkQuota(sub.id, UsageMetric.AgentHours);
      this.addResult({
        category: 'Quota',
        testName: 'Hard Limit Exceeded',
        passed: hardCheck === false,
        details: 'Quota exceeded - operations blocked'
      });
    } catch (error) {
      this.addResult({
        category: 'Quota',
        testName: 'Quota Management',
        passed: false,
        error: String(error)
      });
    }
  }

  private async testIntegration(): Promise<void> {
    console.log('\n🔗 Testing Integration Scenarios...');

    try {
      // Test 1: Full subscription flow with coupon
      const fullFlow = await this.billing.subscribe({
        userId: 'test-integration',
        tier: SubscriptionTier.Pro,
        billingCycle: BillingCycle.Yearly,
        paymentMethodId: 'pm_integration',
        couponCode: 'TEST20'
      });

      this.addResult({
        category: 'Integration',
        testName: 'Full Subscription Flow',
        passed: fullFlow.subscription && fullFlow.payment,
        details: `Created subscription with payment`
      });

      // Test 2: Usage tracking with upgrade
      await this.billing.recordUsage({
        subscriptionId: fullFlow.subscription.id,
        userId: 'test-integration',
        metric: UsageMetric.AgentHours,
        amount: 1800,
        unit: 'hours'
      });

      const beforeUpgrade = await this.billing.getUsageSummary(fullFlow.subscription.id);

      await this.billing.upgrade(fullFlow.subscription.id, SubscriptionTier.Enterprise);

      const afterUpgrade = await this.billing.getUsageSummary(fullFlow.subscription.id);

      this.addResult({
        category: 'Integration',
        testName: 'Usage Persistence After Upgrade',
        passed: beforeUpgrade.metrics.get(UsageMetric.AgentHours) ===
                afterUpgrade.metrics.get(UsageMetric.AgentHours),
        details: 'Usage data persisted after upgrade'
      });

      // Test 3: Event emission
      let eventReceived = false;
      this.billing.once('usage.recorded', () => {
        eventReceived = true;
      });

      await this.billing.recordUsage({
        subscriptionId: fullFlow.subscription.id,
        userId: 'test-integration',
        metric: UsageMetric.APIRequests,
        amount: 5000,
        unit: 'requests'
      });

      // Give event time to fire
      await new Promise(resolve => setTimeout(resolve, 100));

      this.addResult({
        category: 'Integration',
        testName: 'Event System',
        passed: eventReceived === true,
        details: 'Events properly emitted'
      });
    } catch (error) {
      this.addResult({
        category: 'Integration',
        testName: 'Integration Scenarios',
        passed: false,
        error: String(error)
      });
    }
  }

  private addResult(result: TestResult): void {
    this.results.push(result);
    const icon = result.passed ? '✅' : '❌';
    const msg = result.details || result.error || '';
    console.log(`  ${icon} ${result.testName}: ${msg}`);
  }

  private reportResults(): void {
    console.log('\n' + '='.repeat(60));
    console.log('📊 Test Results Summary\n');

    const categories = [...new Set(this.results.map(r => r.category))];
    const totalTests = this.results.length;
    const passedTests = this.results.filter(r => r.passed).length;
    const failedTests = totalTests - passedTests;

    categories.forEach(category => {
      const categoryResults = this.results.filter(r => r.category === category);
      const passed = categoryResults.filter(r => r.passed).length;
      const total = categoryResults.length;
      const percentage = ((passed / total) * 100).toFixed(1);

      console.log(`${category}: ${passed}/${total} (${percentage}%)`);
    });

    console.log('\n' + '='.repeat(60));
    console.log(`Total Tests:    ${totalTests}`);
    console.log(`✅ Passed:      ${passedTests}`);
    console.log(`❌ Failed:      ${failedTests}`);
    console.log(`Success Rate:   ${((passedTests / totalTests) * 100).toFixed(1)}%`);
    console.log('='.repeat(60));

    if (failedTests > 0) {
      console.log('\n⚠️  Failed Tests:');
      this.results.filter(r => !r.passed).forEach(r => {
        console.log(`  - ${r.category}: ${r.testName}`);
        if (r.error) {
          console.log(`    Error: ${r.error}`);
        }
      });
    } else {
      console.log('\n🎉 All tests passed!');
    }
  }
}

// Run tests
const tester = new BillingSystemTest();
tester.runAllTests().catch(console.error);
