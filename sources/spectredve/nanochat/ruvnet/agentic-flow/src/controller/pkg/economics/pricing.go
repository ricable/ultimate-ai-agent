package economics

import (
	"fmt"
)

// PricingManager manages pricing tiers and calculations
type PricingManager struct {
	tiers map[SubscriptionTier]*PricingTier
}

// NewPricingManager creates a new pricing manager
func NewPricingManager() *PricingManager {
	pm := &PricingManager{
		tiers: make(map[SubscriptionTier]*PricingTier),
	}
	pm.initializeDefaultTiers()
	return pm
}

// initializeDefaultTiers sets up the default pricing tiers
func (pm *PricingManager) initializeDefaultTiers() {
	pm.tiers[TierFree] = &PricingTier{
		Tier:         TierFree,
		Name:         "Free",
		Description:  "Perfect for learning and experimentation",
		MonthlyPrice: 0,
		YearlyPrice:  0,
		Currency:     "USD",
		Features: []string{
			"1 concurrent agent",
			"100 agent hours/month",
			"Community support",
			"Basic deployment patterns",
			"1GB storage",
			"Public projects only",
		},
		Limits: &UsageLimits{
			MaxAgentHours:        100,
			MaxDeployments:       10,
			MaxAPIRequests:       10000,
			MaxStorageGB:         1,
			MaxNetworkGB:         10,
			MaxCPUHours:          50,
			MaxMemoryGBHours:     25,
			MaxGPUHours:          0,
			MaxSwarmSize:         1,
			MaxConcurrentBuilds:  1,
			MaxReasoningBankSize: 100, // 100MB
			SupportLevel:         "community",
		},
		OverageRates: map[MetricType]float64{
			MetricAgentHours:  0.10,
			MetricDeployments: 1.00,
			MetricAPIRequests: 0.0001,
			MetricStorageGB:   0.15,
			MetricNetworkGB:   0.05,
		},
	}

	pm.tiers[TierStarter] = &PricingTier{
		Tier:         TierStarter,
		Name:         "Starter",
		Description:  "For individuals and small teams",
		MonthlyPrice: 29,
		YearlyPrice:  290, // 2 months free
		Currency:     "USD",
		Features: []string{
			"3 concurrent agents",
			"500 agent hours/month",
			"Email support (24h response)",
			"All deployment patterns",
			"10GB storage",
			"Private projects",
			"Basic metering dashboard",
			"Coupon support",
		},
		Limits: &UsageLimits{
			MaxAgentHours:        500,
			MaxDeployments:       50,
			MaxAPIRequests:       100000,
			MaxStorageGB:         10,
			MaxNetworkGB:         100,
			MaxCPUHours:          250,
			MaxMemoryGBHours:     125,
			MaxGPUHours:          10,
			MaxSwarmSize:         3,
			MaxConcurrentBuilds:  3,
			MaxReasoningBankSize: 1000, // 1GB
			SupportLevel:         "email",
		},
		OverageRates: map[MetricType]float64{
			MetricAgentHours:  0.08,
			MetricDeployments: 0.75,
			MetricAPIRequests: 0.00008,
			MetricStorageGB:   0.12,
			MetricNetworkGB:   0.04,
			MetricGPUHours:    0.50,
		},
		PopularBadge: false,
	}

	pm.tiers[TierPro] = &PricingTier{
		Tier:         TierPro,
		Name:         "Pro",
		Description:  "For growing businesses and production workloads",
		MonthlyPrice: 99,
		YearlyPrice:  990, // 2 months free
		Currency:     "USD",
		Features: []string{
			"12 concurrent agents",
			"2000 agent hours/month",
			"Priority support (4h response)",
			"All deployment patterns + custom",
			"100GB storage",
			"Private + team projects",
			"Advanced metering & analytics",
			"Cost optimization insights",
			"AI autonomous scaling",
			"ReasoningBank integration",
			"Neural pattern training",
			"Custom coupons",
			"Usage forecasting",
		},
		Limits: &UsageLimits{
			MaxAgentHours:        2000,
			MaxDeployments:       200,
			MaxAPIRequests:       1000000,
			MaxStorageGB:         100,
			MaxNetworkGB:         1000,
			MaxCPUHours:          1000,
			MaxMemoryGBHours:     500,
			MaxGPUHours:          50,
			MaxSwarmSize:         12,
			MaxConcurrentBuilds:  10,
			MaxReasoningBankSize: 10000, // 10GB
			SupportLevel:         "priority",
		},
		OverageRates: map[MetricType]float64{
			MetricAgentHours:  0.06,
			MetricDeployments: 0.50,
			MetricAPIRequests: 0.00005,
			MetricStorageGB:   0.08,
			MetricNetworkGB:   0.03,
			MetricGPUHours:    0.40,
		},
		PopularBadge: true,
	}

	pm.tiers[TierEnterprise] = &PricingTier{
		Tier:         TierEnterprise,
		Name:         "Enterprise",
		Description:  "For large organizations with advanced requirements",
		MonthlyPrice: 499,
		YearlyPrice:  4990, // 2 months free
		Currency:     "USD",
		Features: []string{
			"Unlimited concurrent agents",
			"10000 agent hours/month",
			"24/7 dedicated support",
			"All patterns + unlimited custom",
			"1TB storage",
			"Multi-tenant with SSO",
			"Advanced security (Sigstore, SOC2)",
			"SLA guarantees (99.9% uptime)",
			"Dedicated infrastructure",
			"White-label options",
			"Advanced AI features",
			"Custom neural models",
			"Priority GPU access",
			"Volume discounts",
			"Annual billing discounts",
			"Custom integrations",
			"Training & onboarding",
		},
		Limits: &UsageLimits{
			MaxAgentHours:        10000,
			MaxDeployments:       -1, // unlimited
			MaxAPIRequests:       -1, // unlimited
			MaxStorageGB:         1000,
			MaxNetworkGB:         10000,
			MaxCPUHours:          10000,
			MaxMemoryGBHours:     5000,
			MaxGPUHours:          500,
			MaxSwarmSize:         100,
			MaxConcurrentBuilds:  50,
			MaxReasoningBankSize: 100000, // 100GB
			SupportLevel:         "24x7",
		},
		OverageRates: map[MetricType]float64{
			MetricAgentHours:  0.04,
			MetricDeployments: 0.25,
			MetricAPIRequests: 0.00003,
			MetricStorageGB:   0.05,
			MetricNetworkGB:   0.02,
			MetricGPUHours:    0.30,
		},
		PopularBadge: false,
	}

	pm.tiers[TierCustom] = &PricingTier{
		Tier:          TierCustom,
		Name:          "Custom",
		Description:   "Tailored solutions for unique requirements",
		CustomPricing: true,
		Currency:      "USD",
		Features: []string{
			"Fully customized limits",
			"Negotiated pricing",
			"Custom SLAs",
			"Dedicated account manager",
			"Custom deployment patterns",
			"On-premise options",
			"Air-gapped deployments",
			"Custom compliance requirements",
			"Volume licensing",
			"MSA/DPA agreements",
		},
		Limits: &UsageLimits{
			MaxAgentHours:        -1, // unlimited
			MaxDeployments:       -1,
			MaxAPIRequests:       -1,
			MaxStorageGB:         -1,
			MaxNetworkGB:         -1,
			MaxCPUHours:          -1,
			MaxMemoryGBHours:     -1,
			MaxGPUHours:          -1,
			MaxSwarmSize:         -1,
			MaxConcurrentBuilds:  -1,
			MaxReasoningBankSize: -1,
			SupportLevel:         "custom",
		},
	}
}

// GetTier returns pricing information for a tier
func (pm *PricingManager) GetTier(tier SubscriptionTier) (*PricingTier, error) {
	t, ok := pm.tiers[tier]
	if !ok {
		return nil, fmt.Errorf("tier not found: %s", tier)
	}
	return t, nil
}

// ListTiers returns all available tiers
func (pm *PricingManager) ListTiers() []*PricingTier {
	tiers := make([]*PricingTier, 0, len(pm.tiers))
	// Return in order: Free, Starter, Pro, Enterprise, Custom
	order := []SubscriptionTier{TierFree, TierStarter, TierPro, TierEnterprise, TierCustom}
	for _, tierID := range order {
		if tier, ok := pm.tiers[tierID]; ok {
			tiers = append(tiers, tier)
		}
	}
	return tiers
}

// CalculatePrice calculates the total price including overage
func (pm *PricingManager) CalculatePrice(tier SubscriptionTier, cycle BillingCycle, usage *UsageSummary) (float64, error) {
	pricingTier, err := pm.GetTier(tier)
	if err != nil {
		return 0, err
	}

	// Base price
	var basePrice float64
	switch cycle {
	case CycleMonthly:
		basePrice = pricingTier.MonthlyPrice
	case CycleYearly:
		basePrice = pricingTier.YearlyPrice
	case CycleQuarterly:
		basePrice = pricingTier.YearlyPrice / 4
	default:
		basePrice = pricingTier.MonthlyPrice
	}

	// Calculate overage charges
	overageCharges := 0.0
	if usage != nil {
		for metricType, used := range usage.Metrics {
			limit := pm.getLimitForMetric(pricingTier.Limits, metricType)
			if limit > 0 && used > limit {
				overage := used - limit
				if rate, ok := pricingTier.OverageRates[metricType]; ok {
					overageCharges += overage * rate
				}
			}
		}
	}

	return basePrice + overageCharges, nil
}

// getLimitForMetric returns the limit for a specific metric
func (pm *PricingManager) getLimitForMetric(limits *UsageLimits, metric MetricType) float64 {
	switch metric {
	case MetricAgentHours:
		return float64(limits.MaxAgentHours)
	case MetricDeployments:
		return float64(limits.MaxDeployments)
	case MetricAPIRequests:
		return float64(limits.MaxAPIRequests)
	case MetricStorageGB:
		return limits.MaxStorageGB
	case MetricNetworkGB:
		return limits.MaxNetworkGB
	case MetricCPUHours:
		return limits.MaxCPUHours
	case MetricMemoryGBHours:
		return limits.MaxMemoryGBHours
	case MetricGPUHours:
		return limits.MaxGPUHours
	default:
		return -1 // unlimited
	}
}

// CompareUpgrade compares two tiers for upgrade analysis
func (pm *PricingManager) CompareUpgrade(fromTier, toTier SubscriptionTier) (*UpgradeComparison, error) {
	from, err := pm.GetTier(fromTier)
	if err != nil {
		return nil, err
	}

	to, err := pm.GetTier(toTier)
	if err != nil {
		return nil, err
	}

	return &UpgradeComparison{
		FromTier:       from.Name,
		ToTier:         to.Name,
		PriceDelta:     to.MonthlyPrice - from.MonthlyPrice,
		NewFeatures:    diffFeatures(from.Features, to.Features),
		LimitIncreases: compareLimits(from.Limits, to.Limits),
	}, nil
}

// UpgradeComparison represents a comparison between tiers
type UpgradeComparison struct {
	FromTier       string
	ToTier         string
	PriceDelta     float64
	NewFeatures    []string
	LimitIncreases map[string]string
}

// Helper functions
func diffFeatures(from, to []string) []string {
	fromMap := make(map[string]bool)
	for _, f := range from {
		fromMap[f] = true
	}

	var new []string
	for _, f := range to {
		if !fromMap[f] {
			new = append(new, f)
		}
	}
	return new
}

func compareLimits(from, to *UsageLimits) map[string]string {
	increases := make(map[string]string)

	if to.MaxAgentHours > from.MaxAgentHours {
		increases["Agent Hours"] = fmt.Sprintf("%d → %d", from.MaxAgentHours, to.MaxAgentHours)
	}
	if to.MaxSwarmSize > from.MaxSwarmSize {
		increases["Swarm Size"] = fmt.Sprintf("%d → %d", from.MaxSwarmSize, to.MaxSwarmSize)
	}
	if to.MaxStorageGB > from.MaxStorageGB {
		increases["Storage"] = fmt.Sprintf("%.0fGB → %.0fGB", from.MaxStorageGB, to.MaxStorageGB)
	}
	if to.MaxGPUHours > from.MaxGPUHours {
		increases["GPU Hours"] = fmt.Sprintf("%.0f → %.0f", from.MaxGPUHours, to.MaxGPUHours)
	}

	return increases
}
