# Agentic-Jujutsu TypeScript Billing System

Complete economic system for subscriptions, metering, payments, and access control - native TypeScript implementation for npm.

## 📚 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pricing Tiers](#pricing-tiers)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [CLI Tools](#cli-tools)
- [MCP Integration](#mcp-integration)
- [Storage Adapters](#storage-adapters)
- [API Reference](#api-reference)

## Overview

The TypeScript billing system provides:

- **5 Subscription Tiers** (Free, Starter, Pro, Enterprise, Custom)
- **Real-time Usage Metering** with 10 tracked metrics
- **Quota Enforcement** with soft (80%) and hard (100%) limits
- **Coupon System** with percentage, fixed, and credit discounts
- **Payment Processing** via agentic-payments integration
- **Multiple Storage Backends** (Memory, AgentDB, SQLite, PostgreSQL)
- **Event System** for real-time notifications
- **MCP Tools** for Claude integration

## Installation

```bash
npm install agentic-flow
# or
yarn add agentic-flow
```

Dependencies:
- `agentic-payments` - Payment processing
- `agentdb` - Vector database storage (optional)
- `better-sqlite3` - SQLite storage (optional)

## Quick Start

### Basic Setup

```typescript
import { createBillingSystem, SubscriptionTier, BillingCycle } from 'agentic-flow/billing';

// Create billing system
const billing = createBillingSystem({
  currency: 'USD',
  enableMetering: true,
  enableCoupons: true,
  storageBackend: 'memory'
});

// Create subscription
const { subscription, payment } = await billing.subscribe({
  userId: 'user_123',
  tier: SubscriptionTier.Pro,
  billingCycle: BillingCycle.Monthly,
  paymentMethodId: 'pm_card_123',
  couponCode: 'LAUNCH20' // Optional
});

console.log(`Subscription created: ${subscription.id}`);
console.log(`Monthly cost: $${subscription.price}`);
```

### Record Usage

```typescript
import { UsageMetric } from 'agentic-flow/billing';

// Record agent usage
await billing.recordUsage({
  subscriptionId: subscription.id,
  userId: 'user_123',
  metric: UsageMetric.AgentHours,
  amount: 50,
  unit: 'hours'
});

// Check quota before operation
const allowed = await billing.checkQuota(subscription.id, UsageMetric.AgentHours);
if (!allowed) {
  throw new Error('Quota exceeded');
}
```

### Get Usage Summary

```typescript
const summary = await billing.getUsageSummary(subscription.id);

console.log(`Period: ${summary.period}`);
console.log(`Agent Hours: ${summary.metrics.get(UsageMetric.AgentHours)}/2000`);
console.log(`Percent Used: ${summary.percentUsed.get(UsageMetric.AgentHours)}%`);

if (summary.overages.size > 0) {
  console.log(`Overage Cost: $${summary.estimatedCost}`);
}
```

## Pricing Tiers

### Free Tier - $0/month
Perfect for learning and experimentation

- 100 agent hours/month
- 3 deployments
- 10K API requests
- 1GB storage
- Swarm size up to 3
- Community support

### Starter Tier - $29/month
For small teams and growing projects

- **500 agent hours/month (5x Free)**
- 10 deployments
- 100K API requests
- 10GB storage
- Swarm size up to 6
- 10 GPU hours/month
- Email support
- 1 custom domain

### Pro Tier - $99/month ⭐ POPULAR
For professional teams and production workloads

- **2,000 agent hours/month (20x Free)**
- 50 deployments
- 1M API requests
- 100GB storage
- Swarm size up to 12
- 50 GPU hours/month
- Priority support
- 5 custom domains
- ReasoningBank learning
- QUIC protocol support

### Enterprise Tier - $499/month
For large organizations and mission-critical systems

- **10,000 agent hours/month (100x Free)**
- Unlimited deployments
- 10M API requests
- 1TB storage
- Swarm size up to 50
- 500 GPU hours/month
- 24/7 dedicated support
- Enterprise security (SSO, SAML)
- SLA guarantee (99.9%)
- 50 custom domains
- Byzantine consensus protocols

### Custom Tier - Contact Sales
Tailored solutions for unique requirements

- Unlimited everything
- Custom pricing
- Dedicated infrastructure
- Professional services
- Regulatory compliance support

## Core Components

### 1. Pricing Manager

```typescript
import { PricingManager, SubscriptionTier } from 'agentic-flow/billing';

const pricing = new PricingManager();

// Get tier information
const proTier = pricing.getTier(SubscriptionTier.Pro);
console.log(`Pro: $${proTier.monthlyPrice}/mo`);

// Calculate prices
const yearly = pricing.calculatePrice(SubscriptionTier.Pro, 'yearly');
const savings = pricing.getAnnualSavings(SubscriptionTier.Pro);
console.log(`Yearly: $${yearly} (Save $${savings})`);

// Compare tiers
const comparison = pricing.compareFeatures(
  SubscriptionTier.Starter,
  SubscriptionTier.Pro
);
console.log(`Pro has ${comparison.multipliers.agentHours}x more agent hours`);

// Recommend tier based on usage
const recommended = pricing.getRecommendedTier({
  maxAgentHours: 1500,
  maxStorageGB: 50
});
console.log(`Recommended: ${recommended}`);
```

### 2. Metering Engine

```typescript
import { MeteringEngine, UsageMetric } from 'agentic-flow/billing';

const metering = new MeteringEngine(storage, {
  enabled: true,
  bufferSize: 100,
  flushInterval: 5000,
  softLimitPercent: 80,
  hardLimitPercent: 100
});

// Record usage
await metering.recordUsage({
  subscriptionId: 'sub_123',
  userId: 'user_123',
  metric: UsageMetric.AgentHours,
  amount: 10,
  unit: 'hours'
});

// Check quota
const result = await metering.checkQuota('sub_123', UsageMetric.AgentHours, limits);
if (!result.allowed) {
  console.log(`Quota exceeded: ${result.overage} hours over limit`);
}

// Listen to events
metering.on('quota.warning', (data) => {
  console.log(`Warning: ${data.percentUsed}% of quota used`);
});

metering.on('quota.exceeded', (data) => {
  console.log(`Quota exceeded for ${data.metric}`);
});
```

### 3. Subscription Manager

```typescript
import { SubscriptionManager, SubscriptionTier, BillingCycle } from 'agentic-flow/billing';

const subscriptions = new SubscriptionManager(storage, pricing);

// Create subscription
const sub = await subscriptions.createSubscription({
  userId: 'user_123',
  tier: SubscriptionTier.Starter,
  billingCycle: BillingCycle.Monthly,
  paymentMethodId: 'pm_123'
});

// Upgrade
const upgraded = await subscriptions.upgradeSubscription(sub.id, SubscriptionTier.Pro);
console.log(`Upgraded to ${upgraded.tier}`);

// Downgrade (takes effect at period end)
const downgraded = await subscriptions.downgradeSubscription(sub.id, SubscriptionTier.Starter);

// Cancel
const canceled = await subscriptions.cancelSubscription(sub.id, false); // false = at period end

// Check feature access
const hasAccess = await subscriptions.hasAccess(sub.id, 'ReasoningBank learning');
```

### 4. Coupon Manager

```typescript
import { CouponManager, CouponType, SubscriptionTier } from 'agentic-flow/billing';

const coupons = new CouponManager(storage);

// Create percentage coupon
const percentCoupon = await coupons.createCoupon({
  code: 'LAUNCH20',
  type: CouponType.Percentage,
  value: 20,
  description: '20% off all plans',
  maxRedemptions: 1000,
  validUntil: new Date('2025-12-31'),
  applicableTiers: [SubscriptionTier.Pro, SubscriptionTier.Enterprise]
});

// Create fixed discount
const fixedCoupon = await coupons.createCoupon({
  code: 'SAVE50',
  type: CouponType.Fixed,
  value: 50,
  currency: 'USD',
  minimumAmount: 100
});

// Create credit coupon
const creditCoupon = await coupons.createCoupon({
  code: 'CREDIT100',
  type: CouponType.Credit,
  value: 100,
  description: '$100 account credit'
});

// Validate coupon
const validation = await coupons.validateCoupon('LAUNCH20', SubscriptionTier.Pro, 99);
if (validation.valid) {
  console.log(`Discount: $${validation.discountAmount}`);
  console.log(`Final: $${validation.finalAmount}`);
}

// Apply coupon
await coupons.applyCoupon('LAUNCH20');
```

### 5. Payment Processor

```typescript
import { PaymentProcessor, PaymentProvider } from 'agentic-flow/billing';

const payments = new PaymentProcessor({
  provider: PaymentProvider.Stripe,
  apiKey: process.env.STRIPE_API_KEY,
  testMode: false
}, storage);

// Process payment
const result = await payments.processPayment({
  subscriptionId: 'sub_123',
  userId: 'user_123',
  amount: 99,
  currency: 'USD',
  paymentMethodId: 'pm_card_123',
  description: 'Pro subscription - Monthly'
});

if (result.status === 'succeeded') {
  console.log(`Payment successful: ${result.transactionId}`);
}

// Refund payment
const refund = await payments.refundPayment(result.transactionId, 99, 'Customer request');
```

## Usage Examples

### Example 1: Complete Subscription Flow

```typescript
import { createBillingSystem, SubscriptionTier, BillingCycle, UsageMetric } from 'agentic-flow/billing';

async function subscriptionFlow() {
  const billing = createBillingSystem();

  // 1. Create subscription with coupon
  const { subscription } = await billing.subscribe({
    userId: 'alice_123',
    tier: SubscriptionTier.Pro,
    billingCycle: BillingCycle.Yearly,
    paymentMethodId: 'pm_alice_card',
    couponCode: 'LAUNCH20'
  });

  console.log(`✅ Subscribed to ${subscription.tier}`);
  console.log(`💰 Price: $${subscription.price}/year`);

  // 2. Record usage
  for (let i = 0; i < 10; i++) {
    await billing.recordUsage({
      subscriptionId: subscription.id,
      userId: 'alice_123',
      metric: UsageMetric.AgentHours,
      amount: 50,
      unit: 'hours'
    });
  }

  // 3. Check usage
  const summary = await billing.getUsageSummary(subscription.id);
  const agentHours = summary.metrics.get(UsageMetric.AgentHours);
  const percentUsed = summary.percentUsed.get(UsageMetric.AgentHours);

  console.log(`📊 Used ${agentHours}/2000 hours (${percentUsed.toFixed(1)}%)`);

  // 4. Upgrade if needed
  if (percentUsed > 80) {
    console.log('⚠️  High usage detected, upgrading...');
    await billing.upgrade(subscription.id, SubscriptionTier.Enterprise);
    console.log('✅ Upgraded to Enterprise');
  }

  await billing.shutdown();
}
```

### Example 2: Usage-Based Access Control

```typescript
import { BillingSystem, UsageMetric } from 'agentic-flow/billing';

class AgentService {
  constructor(private billing: BillingSystem) {}

  async executeAgent(subscriptionId: string, hours: number) {
    // Check quota before execution
    const allowed = await this.billing.checkQuota(
      subscriptionId,
      UsageMetric.AgentHours
    );

    if (!allowed) {
      throw new Error('Agent hours quota exceeded. Please upgrade your plan.');
    }

    // Execute agent work...
    console.log(`Executing agent for ${hours} hours`);

    // Record usage
    await this.billing.recordUsage({
      subscriptionId,
      userId: 'system',
      metric: UsageMetric.AgentHours,
      amount: hours,
      unit: 'hours'
    });

    return { success: true, hoursUsed: hours };
  }
}
```

### Example 3: Event-Driven Notifications

```typescript
import { createBillingSystem } from 'agentic-flow/billing';

const billing = createBillingSystem();

// Quota warning notifications
billing.on('quota.warning', async (data) => {
  console.log(`⚠️  ${data.percentUsed}% of ${data.metric} quota used`);

  // Send email notification
  await sendEmail({
    to: data.userId,
    subject: 'Usage Warning',
    body: `You've used ${data.percentUsed}% of your ${data.metric} quota.`
  });
});

// Quota exceeded
billing.on('quota.exceeded', async (data) => {
  console.log(`🚫 Quota exceeded for ${data.metric}`);

  // Block operations and notify
  await blockUserOperations(data.subscriptionId, data.metric);
  await sendEmail({
    to: data.userId,
    subject: 'Quota Exceeded',
    body: `Please upgrade your plan or wait for quota reset.`
  });
});

// Subscription events
billing.on('subscription.upgraded', (data) => {
  console.log(`🎉 Subscription upgraded from ${data.oldTier} to ${data.newTier}`);
});

billing.on('subscription.canceled', (data) => {
  console.log(`😞 Subscription canceled: ${data.subscription.id}`);
});
```

## CLI Tools

### Installation

The billing CLI is included in the agentic-flow package:

```bash
npx agentic-flow billing --help
```

### Available Commands

#### Subscription Management

```bash
# Create subscription
npx agentic-flow billing subscription:create user_123 pro monthly pm_card_123

# Upgrade subscription
npx agentic-flow billing subscription:upgrade sub_123 enterprise

# Cancel subscription
npx agentic-flow billing subscription:cancel sub_123 false

# Get subscription status
npx agentic-flow billing subscription:status sub_123
```

#### Usage Tracking

```bash
# Record usage
npx agentic-flow billing usage:record sub_123 agent_hours 50

# Get usage summary
npx agentic-flow billing usage:summary sub_123

# Check quota
npx agentic-flow billing usage:check sub_123 agent_hours
```

#### Pricing

```bash
# List all tiers
npx agentic-flow billing pricing:tiers

# Compare tiers
npx agentic-flow billing pricing:compare starter pro
```

#### Coupons

```bash
# Create coupon
npx agentic-flow billing coupon:create SAVE20 percentage 20

# Validate coupon
npx agentic-flow billing coupon:validate SAVE20 pro 99

# List coupons
npx agentic-flow billing coupon:list
```

## MCP Integration

Expose billing operations as MCP tools for Claude:

```typescript
import { createBillingSystem, createBillingMCPTools } from 'agentic-flow/billing';
import { FastMCP } from 'fastmcp';

const billing = createBillingSystem();
const mcp = new FastMCP('billing-system');

// Register all billing tools
const billingTools = createBillingMCPTools(billing);
billingTools.getAllTools().forEach(tool => {
  mcp.addTool({
    name: tool.name,
    description: tool.description,
    inputSchema: tool.inputSchema,
    handler: tool.handler
  });
});

// Start MCP server
await mcp.start();
```

### Available MCP Tools

- `billing_subscription_create` - Create subscription
- `billing_subscription_upgrade` - Upgrade tier
- `billing_subscription_cancel` - Cancel subscription
- `billing_subscription_get` - Get subscription details
- `billing_usage_record` - Record usage
- `billing_usage_summary` - Get usage summary
- `billing_quota_check` - Check quota
- `billing_pricing_tiers` - List pricing tiers
- `billing_pricing_calculate` - Calculate price
- `billing_coupon_create` - Create coupon
- `billing_coupon_validate` - Validate coupon

## Storage Adapters

### Memory (Default)

```typescript
import { createBillingSystem } from 'agentic-flow/billing';

const billing = createBillingSystem({
  storageBackend: 'memory'
});
```

### AgentDB (Vector Database)

```typescript
import { createBillingSystem } from 'agentic-flow/billing';
import { AgentDB } from 'agentdb';

const agentdb = new AgentDB({ path: './billing.db' });

const billing = createBillingSystem({
  storageBackend: 'agentdb'
});
```

### SQLite

```typescript
import { createBillingSystem } from 'agentic-flow/billing';

const billing = createBillingSystem({
  storageBackend: 'sqlite'
});
```

### Custom Storage

```typescript
import { StorageAdapter } from 'agentic-flow/billing';

class CustomStorage implements StorageAdapter {
  // Implement all methods...
}

const billing = new BillingSystem({
  storageBackend: 'custom'
});
```

## API Reference

See TypeScript types for complete API documentation:

```typescript
import type {
  // Core types
  Subscription,
  SubscriptionTier,
  BillingCycle,
  SubscriptionStatus,

  // Usage
  UsageRecord,
  UsageMetric,
  UsageSummary,
  UsageLimits,
  QuotaCheckResult,

  // Coupons
  Coupon,
  CouponType,
  CouponValidation,

  // Payments
  PaymentRequest,
  PaymentResult,
  PaymentStatus,
  PaymentProvider,

  // Configuration
  BillingConfig,
  StorageAdapter
} from 'agentic-flow/billing';
```

## Best Practices

1. **Always check quotas** before resource-intensive operations
2. **Handle quota.exceeded events** gracefully
3. **Use yearly billing** for 2 months free savings
4. **Implement proper error handling** for payment failures
5. **Test with memory storage** before production
6. **Monitor usage patterns** for tier recommendations
7. **Set up webhooks** for payment provider events
8. **Cache subscription data** to reduce lookups
9. **Use MCP tools** for Claude integration
10. **Enable event listeners** for real-time notifications

## Support

- **Documentation**: [https://github.com/ruvnet/agentic-flow](https://github.com/ruvnet/agentic-flow)
- **Issues**: [https://github.com/ruvnet/agentic-flow/issues](https://github.com/ruvnet/agentic-flow/issues)
- **Discord**: [Join our community](https://discord.gg/agentic-flow)

---

Built with ❤️ by [@ruvnet](https://github.com/ruvnet)
