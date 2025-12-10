# Agentic-Jujutsu Economic System Guide

Complete guide to the billing, subscriptions, metering, and payment features of agentic-jujutsu.

## Overview

The economic system provides comprehensive value exchange mechanisms including:
- **5 Subscription Tiers**: Free, Starter, Pro, Enterprise, Custom
- **Usage Metering**: 10 tracked metrics with real-time monitoring
- **Flexible Billing**: Monthly, Yearly, Quarterly cycles
- **Coupon System**: Promotional codes and discounts
- **Payment Processing**: Integration with agentic-payments library
- **Quota Enforcement**: Soft and hard limits with access control
- **Overage Billing**: Automatic calculation and charges

---

## Quick Start

```bash
# View pricing tiers
ajj-billing pricing list

# Create subscription
ajj-billing subscription create pro --cycle monthly

# Check current usage
ajj-billing usage current

# Validate coupon
ajj-billing coupon validate WELCOME20

# Check quota status
ajj-billing quota check
```

---

## Subscription Tiers

### 1. Free Tier

**Price**: $0/month

**Perfect for**: Learning, experimentation, hobby projects

**Includes**:
- 1 concurrent agent
- 100 agent hours/month
- 10 deployments/month
- 10,000 API requests/month
- 1GB storage
- 10GB network transfer
- Community support
- Basic deployment patterns
- Public projects only

**Limits**:
- Max swarm size: 1
- No GPU access
- No ReasoningBank
- No custom patterns

### 2. Starter Tier

**Price**: $29/month • $290/year (2 months free)

**Perfect for**: Individuals, small teams, side projects

**Includes**:
- 3 concurrent agents
- 500 agent hours/month
- 50 deployments/month
- 100,000 API requests/month
- 10GB storage
- 100GB network transfer
- 10 GPU hours/month
- Email support (24h response)
- All deployment patterns
- Private projects
- Basic metering dashboard
- Coupon support

**Limits**:
- Max swarm size: 3
- Max concurrent builds: 3
- ReasoningBank: 1GB

**Overage Rates**:
- Agent hours: $0.08/hour
- Deployments: $0.75/deployment
- API requests: $0.00008/request
- Storage: $0.12/GB
- GPU hours: $0.50/hour

### 3. Pro Tier ⭐ POPULAR

**Price**: $99/month • $990/year (2 months free)

**Perfect for**: Growing businesses, production workloads

**Includes**:
- 12 concurrent agents
- 2,000 agent hours/month
- 200 deployments/month
- 1,000,000 API requests/month
- 100GB storage
- 1TB network transfer
- 50 GPU hours/month
- Priority support (4h response)
- All deployment patterns + custom
- Private + team projects
- Advanced metering & analytics
- Cost optimization insights
- AI autonomous scaling
- ReasoningBank integration (10GB)
- Neural pattern training (27 models)
- Custom coupons
- Usage forecasting

**Limits**:
- Max swarm size: 12
- Max concurrent builds: 10

**Overage Rates**:
- Agent hours: $0.06/hour
- Deployments: $0.50/deployment
- API requests: $0.00005/request
- Storage: $0.08/GB
- GPU hours: $0.40/hour

### 4. Enterprise Tier

**Price**: $499/month • $4,990/year (2 months free)

**Perfect for**: Large organizations, mission-critical systems

**Includes**:
- Unlimited concurrent agents
- 10,000 agent hours/month
- Unlimited deployments
- Unlimited API requests
- 1TB storage
- 10TB network transfer
- 500 GPU hours/month
- 24/7 dedicated support
- All patterns + unlimited custom
- Multi-tenant with SSO
- Advanced security (Sigstore, SOC2)
- SLA guarantees (99.9% uptime)
- Dedicated infrastructure
- White-label options
- Advanced AI features
- Custom neural models
- Priority GPU access
- Volume discounts
- Custom integrations
- Training & onboarding

**Limits**:
- Max swarm size: 100
- Max concurrent builds: 50
- ReasoningBank: 100GB

**Overage Rates**:
- Agent hours: $0.04/hour
- Storage: $0.05/GB
- GPU hours: $0.30/hour

### 5. Custom Tier

**Price**: Contact sales

**Perfect for**: Unique requirements, on-premise, air-gapped

**Includes**:
- Fully customized limits
- Negotiated pricing
- Custom SLAs
- Dedicated account manager
- Custom deployment patterns
- On-premise options
- Air-gapped deployments
- Custom compliance requirements
- Volume licensing
- MSA/DPA agreements

---

## Usage Metering

### Tracked Metrics

1. **Agent Hours**: Total hours agents are running
2. **Deployments**: Number of deployments per month
3. **API Requests**: Total API calls
4. **Storage (GB)**: Total storage used
5. **Network (GB)**: Data transfer per month
6. **CPU Hours**: Compute time
7. **Memory GB-Hours**: Memory usage over time
8. **GPU Hours**: GPU compute time
9. **Swarm Coordination**: Multi-agent coordination events
10. **Neural Training**: Neural model training sessions

### Real-Time Tracking

```bash
# View current usage
ajj-billing usage current

# Output:
# Current Usage (November 2025):
#
# Agent Hours:       450 / 2000  (22.5%)
# Deployments:       35 / 200    (17.5%)
# API Requests:      250K / 1M   (25.0%)
# Storage:           45GB / 100GB (45.0%)
# Network:           180GB / 1TB  (18.0%)
# GPU Hours:         12 / 50     (24.0%)
#
# Estimated Cost: $99.00
# Overage Charges: $0.00
# Total: $99.00
```

### Usage History

```bash
# View historical usage
ajj-billing usage history

# Forecast future usage
ajj-billing usage forecast
```

---

## Coupon System

### Creating Coupons

```bash
# Create percentage discount
ajj-billing coupon create WELCOME20 --type percentage --value 20

# Create fixed discount
ajj-billing coupon create STARTUP50 --type fixed --value 50

# Create credits coupon
ajj-billing coupon create CREDITS100 --type credits --value 100
```

### Coupon Types

1. **Percentage**: Discount as percentage (e.g., 20% off)
2. **Fixed**: Fixed amount discount (e.g., $50 off)
3. **Credits**: Account credits (e.g., $100 credit)

### Coupon Features

- ✅ **Validity Period**: Set start and end dates
- ✅ **Usage Limits**: Maximum number of uses
- ✅ **First-Time Only**: Restrict to new subscribers
- ✅ **Tier Restrictions**: Apply to specific tiers
- ✅ **Minimum Purchase**: Require minimum amount
- ✅ **Stackable**: Combine multiple coupons

### Applying Coupons

```bash
# Validate coupon
ajj-billing coupon validate WELCOME20

# Create subscription with coupon
ajj-billing subscription create pro --coupon WELCOME20

# Result:
# Base price: $99.00
# Discount (WELCOME20): -$19.80
# Final price: $79.20
```

---

## Quota Management

### Quota Levels

1. **Soft Limit (80%)**: Warning notification sent
2. **Hard Limit (100%)**: Operations blocked

### Checking Quotas

```bash
# Check all quotas
ajj-billing quota check

# Output:
# Quota Status:
#
# ✓ Agent Hours    : 450 / 2000 hours (22.5%)
# ✓ Deployments    : 35 / 200 deployments (17.5%)
# ✓ API Requests   : 250000 / 1000000 requests (25.0%)
# ✓ Storage        : 45 / 100 GB (45.0%)
# ✓ Network        : 180 / 1000 GB (18.0%)
# ⚠ GPU Hours      : 48 / 50 hours (96.0%)  # Warning!
#
# Status: 1 quota nearing limit
```

### Quota Enforcement

- **80% (Soft Limit)**: Email notification sent
- **90%**: Warning in dashboard
- **95%**: Urgent notification
- **100% (Hard Limit)**: Operations blocked, upgrade prompt shown

### Requesting Quota Increase

```bash
ajj-billing quota increase

# Suggests upgrade path:
# Current: Pro ($99/month)
# Upgrade to: Enterprise ($499/month)
#
# Increase benefits:
#   • Agent Hours: 2000 → 10000
#   • Concurrent Agents: 12 → 100
#   • Storage: 100GB → 1TB
#   • GPU Hours: 50 → 500
```

---

## Payment Processing

### Supported Payment Methods

1. **Credit/Debit Cards**: Visa, Mastercard, Amex, Discover
2. **Bank Accounts**: ACH, wire transfer
3. **Crypto**: Bitcoin, Ethereum, USDC

### Payment Providers

- **Stripe**: Primary card processing
- **PayPal**: Alternative payment method
- **Crypto**: Blockchain payments

### Adding Payment Method

```bash
# Via CLI (opens secure form)
ajj-billing payment-method add --type card

# Via API
curl -X POST /api/payment-methods \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"type": "card", "token": "tok_..."}'
```

### Payment Features

- ✅ **Auto-Retry**: Failed payments retried 3 times
- ✅ **Dunning**: Progressive reminder emails
- ✅ **PCI Compliance**: Secure card handling
- ✅ **3D Secure**: Enhanced fraud protection
- ✅ **Refunds**: Processed within 5-10 business days
- ✅ **Receipts**: Automatic email receipts
- ✅ **Invoices**: Monthly detailed invoices

---

## Billing Cycles

### Monthly Billing

- Charged on the same day each month
- Prorated charges for mid-cycle changes
- 7-day renewal reminder

### Yearly Billing

- **Savings**: 2 months free (16.7% discount)
- Charged annually
- Prorated refunds for downgrades

### Quarterly Billing

- Charged every 3 months
- Balance between monthly and yearly

### Billing Examples

**Monthly**: $99/month = $1,188/year
**Yearly**: $990/year = $82.50/month (saves $198)

---

## Subscription Management

### Creating Subscription

```bash
# Create new subscription
ajj-billing subscription create pro --cycle monthly

# With payment method
ajj-billing subscription create pro \
  --cycle yearly \
  --payment-method pm_abc123

# With coupon
ajj-billing subscription create pro \
  --cycle monthly \
  --coupon WELCOME20
```

### Upgrading Subscription

```bash
# Upgrade to higher tier
ajj-billing subscription upgrade enterprise

# Process:
# 1. Calculate prorated amount
# 2. Charge difference
# 3. Update subscription immediately
# 4. Apply new limits
```

**Prorated Billing Example**:
- Current: Pro $99/month (15 days remaining)
- Upgrade to: Enterprise $499/month
- Prorated charge: ($499 - $99) × (15/30) = $200

### Downgrading Subscription

```bash
# Downgrade to lower tier
ajj-billing subscription downgrade starter

# Process:
# 1. Schedule downgrade for end of period
# 2. Continue current tier until renewal
# 3. Apply new tier and limits at renewal
```

### Cancelling Subscription

```bash
# Cancel at end of period (recommended)
ajj-billing subscription cancel

# Cancel immediately
ajj-billing subscription cancel --immediate

# Process:
# 1. Stop recurring charges
# 2. Access continues until period end
# 3. Data retained for 30 days
# 4. Reactivation available within 30 days
```

### Checking Status

```bash
ajj-billing subscription status

# Output:
# Subscription Status:
#   Tier: Pro
#   Status: Active
#   Billing Cycle: Monthly
#   Next Billing Date: 2025-12-15
#   Price: $99.00/month
#   Features: 12 concurrent agents, 2000 agent hours
```

---

## Integration with agentic-payments

The economic system integrates with the **agentic-payments** library (v0.1.13) for payment processing.

### Features

- ✅ Multi-provider support (Stripe, PayPal, Crypto)
- ✅ Webhook handling
- ✅ Payment retry logic
- ✅ Fraud detection
- ✅ Multi-currency support
- ✅ Subscription management
- ✅ Invoice generation
- ✅ Refund processing

### Usage

```typescript
import { AgenticPayments } from 'agentic-payments';

const payments = new AgenticPayments({
  provider: 'stripe',
  apiKey: process.env.STRIPE_KEY,
});

// Process payment
const result = await payments.charge({
  amount: 9900,
  currency: 'usd',
  paymentMethod: 'pm_abc123',
});
```

---

## API Reference

### Pricing Endpoints

```bash
GET    /api/pricing/tiers          # List all tiers
GET    /api/pricing/tiers/:tier    # Get tier details
POST   /api/pricing/calculate      # Calculate price
```

### Subscription Endpoints

```bash
POST   /api/subscriptions          # Create subscription
GET    /api/subscriptions/:id      # Get subscription
PATCH  /api/subscriptions/:id      # Update subscription
DELETE /api/subscriptions/:id      # Cancel subscription
POST   /api/subscriptions/:id/upgrade   # Upgrade
```

### Usage Endpoints

```bash
POST   /api/usage/record           # Record usage
GET    /api/usage/current          # Current usage
GET    /api/usage/history          # Historical usage
GET    /api/usage/forecast         # Usage forecast
```

### Coupon Endpoints

```bash
POST   /api/coupons                # Create coupon
GET    /api/coupons/:code          # Get coupon
POST   /api/coupons/:code/validate # Validate coupon
```

### Payment Endpoints

```bash
POST   /api/payments               # Process payment
POST   /api/payments/:id/refund    # Refund payment
GET    /api/payment-methods        # List methods
POST   /api/payment-methods        # Add method
```

---

## Best Practices

### For Users

1. **Start Small**: Begin with Free or Starter tier
2. **Use Coupons**: Check for promotional codes
3. **Monitor Usage**: Review dashboard regularly
4. **Set Alerts**: Configure quota warnings
5. **Annual Billing**: Save 2 months with yearly plans
6. **Review Regularly**: Downgrade if underutilizing

### For Developers

1. **Implement Metering**: Track all billable events
2. **Handle Webhooks**: Process payment events
3. **Grace Periods**: Allow time before hard limits
4. **Clear Communication**: Show usage and limits
5. **Easy Upgrades**: Make it simple to increase limits
6. **Transparent Pricing**: No hidden fees

---

## Troubleshooting

### Payment Failed

1. Check payment method validity
2. Verify sufficient funds
3. Try alternative payment method
4. Contact support if persistent

### Quota Exceeded

1. Check current usage: `ajj-billing usage current`
2. Review quota status: `ajj-billing quota check`
3. Upgrade tier: `ajj-billing subscription upgrade`
4. Contact sales for custom limits

### Coupon Not Working

1. Validate coupon: `ajj-billing coupon validate CODE`
2. Check validity period
3. Verify tier applicability
4. Check usage limits
5. Ensure minimum purchase met

### Subscription Issues

1. Check status: `ajj-billing subscription status`
2. Verify payment method
3. Review billing history
4. Contact support with subscription ID

---

## FAQ

**Q: Can I change billing cycles?**
A: Yes, you can switch between monthly, yearly, and quarterly at any time.

**Q: What happens if I exceed my limits?**
A: You'll be charged overage rates. Consider upgrading to avoid charges.

**Q: Can I get a refund?**
A: Yearly subscriptions are prorated for downgrades. Contact support for special cases.

**Q: Do unused hours roll over?**
A: No, limits reset each billing period.

**Q: Can I pause my subscription?**
A: Yes, contact support to pause for up to 3 months.

**Q: Is there a free trial?**
A: The Free tier is available indefinitely. Paid tiers offer 14-day money-back guarantee.

**Q: Can I use multiple coupons?**
A: Generally one per subscription, but some can be stacked. Check coupon details.

**Q: What payment methods do you accept?**
A: Credit/debit cards, bank accounts, and cryptocurrency.

---

## Support

For billing questions or issues:
- **Email**: billing@agentic-jujutsu.com
- **Documentation**: /docs/economics
- **CLI Help**: `ajj-billing --help`
- **API Reference**: /api/docs

---

**Version**: 1.0.0
**Last Updated**: 2025-11-16
**Status**: Production Ready ✅
