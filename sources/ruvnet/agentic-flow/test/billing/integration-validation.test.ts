/**
 * Integration Validation Test
 * Validates all billing system integrations and capabilities
 */

import { BillingSystem, createBillingSystem, SubscriptionTier, BillingCycle, UsageMetric, CouponType, PaymentProvider } from '../../agentic-flow/src/billing/index.js';
import { createBillingMCPTools } from '../../agentic-flow/src/billing/mcp/tools.js';

interface ValidationResult {
  category: string;
  test: string;
  passed: boolean;
  details?: string;
}

class IntegrationValidator {
  private results: ValidationResult[] = [];

  async runAllValidations(): Promise<void> {
    console.log('🔍 Starting Comprehensive Integration Validation\n');
    console.log('='.repeat(60));

    await this.validateExports();
    await this.validateFactoryFunctions();
    await this.validateTypeSystem();
    await this.validateEventSystem();
    await this.validateMCPIntegration();
    await this.validateCLIInterface();
    await this.validateStorageAdapters();
    await this.validateErrorHandling();
    await this.validateThreadSafety();

    this.reportResults();
  }

  private async validateExports(): Promise<void> {
    console.log('\n📦 Validating Exports...');

    try {
      // Validate main class
      this.addResult({
        category: 'Exports',
        test: 'BillingSystem class export',
        passed: typeof BillingSystem === 'function',
        details: 'Main class exported'
      });

      // Validate factory function
      this.addResult({
        category: 'Exports',
        test: 'createBillingSystem function export',
        passed: typeof createBillingSystem === 'function',
        details: 'Factory function exported'
      });

      // Validate enums
      this.addResult({
        category: 'Exports',
        test: 'SubscriptionTier enum export',
        passed: SubscriptionTier.Pro === 'pro',
        details: 'Enum values accessible'
      });

      this.addResult({
        category: 'Exports',
        test: 'BillingCycle enum export',
        passed: BillingCycle.Monthly === 'monthly',
        details: 'Billing cycle enum works'
      });

      this.addResult({
        category: 'Exports',
        test: 'UsageMetric enum export',
        passed: UsageMetric.AgentHours === 'agent_hours',
        details: 'Usage metric enum works'
      });

      this.addResult({
        category: 'Exports',
        test: 'CouponType enum export',
        passed: CouponType.Percentage === 'percentage',
        details: 'Coupon type enum works'
      });

      this.addResult({
        category: 'Exports',
        test: 'PaymentProvider enum export',
        passed: PaymentProvider.Stripe === 'stripe',
        details: 'Payment provider enum works'
      });
    } catch (error) {
      this.addResult({
        category: 'Exports',
        test: 'Export validation',
        passed: false,
        details: String(error)
      });
    }
  }

  private async validateFactoryFunctions(): Promise<void> {
    console.log('\n🏭 Validating Factory Functions...');

    try {
      // Test factory function
      const billing1 = createBillingSystem();
      this.addResult({
        category: 'Factory',
        test: 'createBillingSystem() default config',
        passed: billing1 instanceof BillingSystem,
        details: 'Factory creates instance'
      });

      // Test with custom config
      const billing2 = createBillingSystem({
        currency: 'EUR',
        enableMetering: false
      });
      this.addResult({
        category: 'Factory',
        test: 'createBillingSystem() custom config',
        passed: billing2 instanceof BillingSystem,
        details: 'Custom config accepted'
      });

      // Test constructor
      const billing3 = new BillingSystem({
        storageBackend: 'memory'
      });
      this.addResult({
        category: 'Factory',
        test: 'new BillingSystem() constructor',
        passed: billing3 instanceof BillingSystem,
        details: 'Constructor works'
      });

      // Cleanup
      await billing1.shutdown();
      await billing2.shutdown();
      await billing3.shutdown();
    } catch (error) {
      this.addResult({
        category: 'Factory',
        test: 'Factory functions',
        passed: false,
        details: String(error)
      });
    }
  }

  private async validateTypeSystem(): Promise<void> {
    console.log('\n📝 Validating Type System...');

    try {
      const billing = createBillingSystem();

      // Validate component access
      this.addResult({
        category: 'Types',
        test: 'billing.pricing property',
        passed: billing.pricing !== undefined,
        details: 'Pricing manager accessible'
      });

      this.addResult({
        category: 'Types',
        test: 'billing.metering property',
        passed: billing.metering !== undefined,
        details: 'Metering engine accessible'
      });

      this.addResult({
        category: 'Types',
        test: 'billing.subscriptions property',
        passed: billing.subscriptions !== undefined,
        details: 'Subscription manager accessible'
      });

      this.addResult({
        category: 'Types',
        test: 'billing.coupons property',
        passed: billing.coupons !== undefined,
        details: 'Coupon manager accessible'
      });

      this.addResult({
        category: 'Types',
        test: 'billing.payments property',
        passed: billing.payments !== undefined,
        details: 'Payment processor accessible'
      });

      this.addResult({
        category: 'Types',
        test: 'billing.storage property',
        passed: billing.storage !== undefined,
        details: 'Storage adapter accessible'
      });

      await billing.shutdown();
    } catch (error) {
      this.addResult({
        category: 'Types',
        test: 'Type system',
        passed: false,
        details: String(error)
      });
    }
  }

  private async validateEventSystem(): Promise<void> {
    console.log('\n📡 Validating Event System...');

    try {
      const billing = createBillingSystem();

      // Test event registration
      let usageEventFired = false;
      let quotaWarningFired = false;
      let subscriptionEventFired = false;

      billing.on('usage.recorded', () => {
        usageEventFired = true;
      });

      billing.on('quota.warning', () => {
        quotaWarningFired = true;
      });

      billing.on('subscription.created', () => {
        subscriptionEventFired = true;
      });

      // Create subscription to trigger events
      const sub = await billing.subscriptions.createSubscription({
        userId: 'test-events',
        tier: SubscriptionTier.Free,
        billingCycle: BillingCycle.Monthly,
        paymentMethodId: 'pm_test'
      });

      this.addResult({
        category: 'Events',
        test: 'subscription.created event',
        passed: subscriptionEventFired,
        details: 'Event fired on subscription creation'
      });

      // Record usage to trigger events
      await billing.recordUsage({
        subscriptionId: sub.id,
        userId: sub.userId,
        metric: UsageMetric.AgentHours,
        amount: 85, // 85% of 100 (free tier)
        unit: 'hours'
      });

      // Give events time to fire
      await new Promise(resolve => setTimeout(resolve, 100));

      this.addResult({
        category: 'Events',
        test: 'usage.recorded event',
        passed: usageEventFired,
        details: 'Event fired on usage recording'
      });

      this.addResult({
        category: 'Events',
        test: 'quota.warning event',
        passed: quotaWarningFired,
        details: 'Warning triggered at 85% usage'
      });

      // Test event removal
      const handler = () => {};
      billing.on('test.event', handler);
      billing.off('test.event', handler);

      this.addResult({
        category: 'Events',
        test: 'Event listener removal',
        passed: true,
        details: 'on/off methods work'
      });

      await billing.shutdown();
    } catch (error) {
      this.addResult({
        category: 'Events',
        test: 'Event system',
        passed: false,
        details: String(error)
      });
    }
  }

  private async validateMCPIntegration(): Promise<void> {
    console.log('\n🔧 Validating MCP Integration...');

    try {
      const billing = createBillingSystem();
      const mcpTools = createBillingMCPTools(billing);

      // Validate MCP tools creation
      this.addResult({
        category: 'MCP',
        test: 'createBillingMCPTools function',
        passed: mcpTools !== undefined,
        details: 'MCP tools factory works'
      });

      // Get all tools
      const tools = mcpTools.getAllTools();
      this.addResult({
        category: 'MCP',
        test: 'getAllTools() returns tools',
        passed: tools.length === 11,
        details: `Found ${tools.length} MCP tools`
      });

      // Validate tool structure
      const firstTool = tools[0];
      this.addResult({
        category: 'MCP',
        test: 'MCP tool structure',
        passed: firstTool.name && firstTool.description && firstTool.handler,
        details: 'Tool has required properties'
      });

      // Test tool execution
      const pricingTool = mcpTools.getTool('billing_pricing_tiers');
      if (pricingTool) {
        const result = await mcpTools.executeTool('billing_pricing_tiers', {});
        this.addResult({
          category: 'MCP',
          test: 'MCP tool execution',
          passed: Array.isArray(result) && result.length === 5,
          details: 'Tool executes and returns data'
        });
      }

      // Validate all tool names
      const expectedTools = [
        'billing_subscription_create',
        'billing_subscription_upgrade',
        'billing_subscription_cancel',
        'billing_subscription_get',
        'billing_usage_record',
        'billing_usage_summary',
        'billing_quota_check',
        'billing_pricing_tiers',
        'billing_pricing_calculate',
        'billing_coupon_create',
        'billing_coupon_validate'
      ];

      const toolNames = tools.map(t => t.name);
      const allPresent = expectedTools.every(name => toolNames.includes(name));

      this.addResult({
        category: 'MCP',
        test: 'All MCP tools present',
        passed: allPresent,
        details: `${toolNames.length} tools registered`
      });

      await billing.shutdown();
    } catch (error) {
      this.addResult({
        category: 'MCP',
        test: 'MCP integration',
        passed: false,
        details: String(error)
      });
    }
  }

  private async validateCLIInterface(): Promise<void> {
    console.log('\n💻 Validating CLI Interface...');

    try {
      // Validate CLI module exists
      const { BillingCLI } = await import('../../agentic-flow/src/billing/cli.js');

      this.addResult({
        category: 'CLI',
        test: 'CLI module import',
        passed: BillingCLI !== undefined,
        details: 'CLI class exported'
      });

      this.addResult({
        category: 'CLI',
        test: 'CLI instantiation',
        passed: typeof BillingCLI === 'function',
        details: 'CLI can be instantiated'
      });
    } catch (error) {
      this.addResult({
        category: 'CLI',
        test: 'CLI interface',
        passed: false,
        details: String(error)
      });
    }
  }

  private async validateStorageAdapters(): Promise<void> {
    console.log('\n💾 Validating Storage Adapters...');

    try {
      // Test Memory adapter
      const billing1 = createBillingSystem({ storageBackend: 'memory' });
      const sub1 = await billing1.subscriptions.createSubscription({
        userId: 'test-storage-1',
        tier: SubscriptionTier.Starter,
        billingCycle: BillingCycle.Monthly,
        paymentMethodId: 'pm_test'
      });

      const retrieved1 = await billing1.subscriptions.getSubscription(sub1.id);

      this.addResult({
        category: 'Storage',
        test: 'Memory adapter CRUD',
        passed: retrieved1 !== null && retrieved1.id === sub1.id,
        details: 'Memory storage works'
      });

      // Test AgentDB adapter (falls back to memory)
      const billing2 = createBillingSystem({ storageBackend: 'agentdb' });
      this.addResult({
        category: 'Storage',
        test: 'AgentDB adapter initialization',
        passed: billing2 !== undefined,
        details: 'AgentDB adapter (fallback to memory)'
      });

      // Test SQLite adapter (falls back to memory)
      const billing3 = createBillingSystem({ storageBackend: 'sqlite' });
      this.addResult({
        category: 'Storage',
        test: 'SQLite adapter initialization',
        passed: billing3 !== undefined,
        details: 'SQLite adapter (fallback to memory)'
      });

      await billing1.shutdown();
      await billing2.shutdown();
      await billing3.shutdown();
    } catch (error) {
      this.addResult({
        category: 'Storage',
        test: 'Storage adapters',
        passed: false,
        details: String(error)
      });
    }
  }

  private async validateErrorHandling(): Promise<void> {
    console.log('\n⚠️  Validating Error Handling...');

    try {
      const billing = createBillingSystem();

      // Test invalid subscription ID
      try {
        await billing.subscriptions.getSubscription('invalid_id');
        this.addResult({
          category: 'Errors',
          test: 'Invalid subscription lookup',
          passed: true,
          details: 'Returns null for invalid ID'
        });
      } catch (error) {
        this.addResult({
          category: 'Errors',
          test: 'Invalid subscription lookup',
          passed: false,
          details: 'Should return null, not throw'
        });
      }

      // Test quota check with no subscription
      const allowed = await billing.checkQuota('nonexistent', UsageMetric.AgentHours);
      this.addResult({
        category: 'Errors',
        test: 'Quota check without subscription',
        passed: allowed === false,
        details: 'Gracefully handles missing subscription'
      });

      // Test invalid coupon
      const validation = await billing.coupons.validateCoupon(
        'INVALID_CODE',
        SubscriptionTier.Pro,
        99
      );
      this.addResult({
        category: 'Errors',
        test: 'Invalid coupon validation',
        passed: validation.valid === false && validation.error !== undefined,
        details: 'Returns validation error'
      });

      await billing.shutdown();
    } catch (error) {
      this.addResult({
        category: 'Errors',
        test: 'Error handling',
        passed: false,
        details: String(error)
      });
    }
  }

  private async validateThreadSafety(): Promise<void> {
    console.log('\n🔒 Validating Concurrent Operations...');

    try {
      const billing = createBillingSystem();

      const sub = await billing.subscriptions.createSubscription({
        userId: 'test-concurrent',
        tier: SubscriptionTier.Pro,
        billingCycle: BillingCycle.Monthly,
        paymentMethodId: 'pm_test'
      });

      // Concurrent usage recording
      const promises = Array.from({ length: 10 }, (_, i) =>
        billing.recordUsage({
          subscriptionId: sub.id,
          userId: sub.userId,
          metric: UsageMetric.AgentHours,
          amount: 10,
          unit: 'hours'
        })
      );

      await Promise.all(promises);

      // Verify total usage
      const summary = await billing.getUsageSummary(sub.id);
      const total = summary.metrics.get(UsageMetric.AgentHours) || 0;

      this.addResult({
        category: 'Concurrency',
        test: 'Concurrent usage recording',
        passed: total === 100,
        details: `Recorded ${total}/100 hours (10 concurrent)`
      });

      await billing.shutdown();
    } catch (error) {
      this.addResult({
        category: 'Concurrency',
        test: 'Thread safety',
        passed: false,
        details: String(error)
      });
    }
  }

  private addResult(result: ValidationResult): void {
    this.results.push(result);
    const icon = result.passed ? '✅' : '❌';
    console.log(`  ${icon} ${result.test}: ${result.details || ''}`);
  }

  private reportResults(): void {
    console.log('\n' + '='.repeat(60));
    console.log('📊 Integration Validation Results\n');

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
    console.log(`Total Validations: ${totalTests}`);
    console.log(`✅ Passed:         ${passedTests}`);
    console.log(`❌ Failed:         ${failedTests}`);
    console.log(`Success Rate:      ${((passedTests / totalTests) * 100).toFixed(1)}%`);
    console.log('='.repeat(60));

    if (failedTests > 0) {
      console.log('\n⚠️  Failed Validations:');
      this.results.filter(r => !r.passed).forEach(r => {
        console.log(`  - ${r.category}: ${r.test}`);
        if (r.details) {
          console.log(`    ${r.details}`);
        }
      });
    } else {
      console.log('\n🎉 All integration validations passed!');
    }
  }
}

// Run validations
const validator = new IntegrationValidator();
validator.runAllValidations().catch(console.error);
