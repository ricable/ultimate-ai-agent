package economics

import (
	"time"
)

// SubscriptionTier represents different subscription levels
type SubscriptionTier string

const (
	TierFree       SubscriptionTier = "free"
	TierStarter    SubscriptionTier = "starter"
	TierPro        SubscriptionTier = "pro"
	TierEnterprise SubscriptionTier = "enterprise"
	TierCustom     SubscriptionTier = "custom"
)

// BillingCycle represents billing frequency
type BillingCycle string

const (
	CycleMonthly  BillingCycle = "monthly"
	CycleYearly   BillingCycle = "yearly"
	CycleQuarterly BillingCycle = "quarterly"
)

// MetricType represents different types of usage metrics
type MetricType string

const (
	MetricAgentHours        MetricType = "agent_hours"
	MetricDeployments       MetricType = "deployments"
	MetricAPIRequests       MetricType = "api_requests"
	MetricStorageGB         MetricType = "storage_gb"
	MetricNetworkGB         MetricType = "network_gb"
	MetricCPUHours          MetricType = "cpu_hours"
	MetricMemoryGBHours     MetricType = "memory_gb_hours"
	MetricGPUHours          MetricType = "gpu_hours"
	MetricSwarmCoordination MetricType = "swarm_coordination"
	MetricNeuralTraining    MetricType = "neural_training"
)

// Subscription represents a customer subscription
type Subscription struct {
	ID              string           `json:"id"`
	UserID          string           `json:"user_id"`
	Tier            SubscriptionTier `json:"tier"`
	BillingCycle    BillingCycle     `json:"billing_cycle"`
	Status          string           `json:"status"` // active, paused, cancelled, expired
	StartDate       time.Time        `json:"start_date"`
	EndDate         time.Time        `json:"end_date"`
	RenewalDate     time.Time        `json:"renewal_date"`
	Price           float64          `json:"price"`
	Currency        string           `json:"currency"`
	Features        []string         `json:"features"`
	Limits          *UsageLimits     `json:"limits"`
	PaymentMethodID string           `json:"payment_method_id"`
	Metadata        map[string]string `json:"metadata"`
}

// UsageLimits represents quota limits for a subscription
type UsageLimits struct {
	MaxAgentHours        int     `json:"max_agent_hours"`        // per month
	MaxDeployments       int     `json:"max_deployments"`        // per month
	MaxAPIRequests       int     `json:"max_api_requests"`       // per month
	MaxStorageGB         float64 `json:"max_storage_gb"`         // total
	MaxNetworkGB         float64 `json:"max_network_gb"`         // per month
	MaxCPUHours          float64 `json:"max_cpu_hours"`          // per month
	MaxMemoryGBHours     float64 `json:"max_memory_gb_hours"`    // per month
	MaxGPUHours          float64 `json:"max_gpu_hours"`          // per month
	MaxSwarmSize         int     `json:"max_swarm_size"`         // concurrent agents
	MaxConcurrentBuilds  int     `json:"max_concurrent_builds"`  // parallel builds
	MaxReasoningBankSize int     `json:"max_reasoning_bank_size"` // MB
	SupportLevel         string  `json:"support_level"`          // community, email, priority, 24x7
}

// UsageRecord represents actual usage data
type UsageRecord struct {
	ID          string     `json:"id"`
	UserID      string     `json:"user_id"`
	Timestamp   time.Time  `json:"timestamp"`
	MetricType  MetricType `json:"metric_type"`
	Value       float64    `json:"value"`
	Unit        string     `json:"unit"`
	ResourceID  string     `json:"resource_id"`
	Cost        float64    `json:"cost"`
	BillingPeriod string   `json:"billing_period"` // 2025-01, 2025-Q1
	Metadata    map[string]string `json:"metadata"`
}

// UsageSummary represents aggregated usage for a period
type UsageSummary struct {
	UserID        string               `json:"user_id"`
	Period        string               `json:"period"`
	Tier          SubscriptionTier     `json:"tier"`
	TotalCost     float64              `json:"total_cost"`
	Currency      string               `json:"currency"`
	Metrics       map[MetricType]float64 `json:"metrics"`
	Limits        *UsageLimits         `json:"limits"`
	PercentUsed   map[MetricType]float64 `json:"percent_used"`
	OverageCharges float64             `json:"overage_charges"`
	Credits       float64              `json:"credits"`
	NetAmount     float64              `json:"net_amount"`
}

// Coupon represents a promotional discount
type Coupon struct {
	ID               string    `json:"id"`
	Code             string    `json:"code"`
	Description      string    `json:"description"`
	Type             string    `json:"type"` // percentage, fixed, credits
	Value            float64   `json:"value"`
	Currency         string    `json:"currency,omitempty"`
	MaxUses          int       `json:"max_uses"`
	UsedCount        int       `json:"used_count"`
	ValidFrom        time.Time `json:"valid_from"`
	ValidUntil       time.Time `json:"valid_until"`
	MinimumPurchase  float64   `json:"minimum_purchase,omitempty"`
	ApplicableTiers  []SubscriptionTier `json:"applicable_tiers,omitempty"`
	FirstTimeOnly    bool      `json:"first_time_only"`
	Status           string    `json:"status"` // active, expired, disabled
	Metadata         map[string]string `json:"metadata"`
}

// Invoice represents a billing invoice
type Invoice struct {
	ID              string            `json:"id"`
	UserID          string            `json:"user_id"`
	SubscriptionID  string            `json:"subscription_id"`
	InvoiceNumber   string            `json:"invoice_number"`
	Status          string            `json:"status"` // draft, open, paid, void, uncollectible
	CreatedAt       time.Time         `json:"created_at"`
	DueDate         time.Time         `json:"due_date"`
	PaidAt          *time.Time        `json:"paid_at,omitempty"`
	Period          string            `json:"period"`
	Subtotal        float64           `json:"subtotal"`
	Discount        float64           `json:"discount"`
	Tax             float64           `json:"tax"`
	Total           float64           `json:"total"`
	Currency        string            `json:"currency"`
	LineItems       []InvoiceLineItem `json:"line_items"`
	CouponID        string            `json:"coupon_id,omitempty"`
	PaymentMethodID string            `json:"payment_method_id,omitempty"`
	Metadata        map[string]string `json:"metadata"`
}

// InvoiceLineItem represents a line item on an invoice
type InvoiceLineItem struct {
	Description string  `json:"description"`
	Quantity    float64 `json:"quantity"`
	UnitPrice   float64 `json:"unit_price"`
	Amount      float64 `json:"amount"`
	MetricType  MetricType `json:"metric_type,omitempty"`
}

// PaymentMethod represents a payment method
type PaymentMethod struct {
	ID          string    `json:"id"`
	UserID      string    `json:"user_id"`
	Type        string    `json:"type"` // card, bank_account, crypto
	Provider    string    `json:"provider"` // stripe, paypal, crypto
	Last4       string    `json:"last4"`
	Brand       string    `json:"brand"`
	ExpiryMonth int       `json:"expiry_month,omitempty"`
	ExpiryYear  int       `json:"expiry_year,omitempty"`
	IsDefault   bool      `json:"is_default"`
	Status      string    `json:"status"` // active, expired, failed
	CreatedAt   time.Time `json:"created_at"`
	Metadata    map[string]string `json:"metadata"`
}

// PricingTier represents pricing configuration for a tier
type PricingTier struct {
	Tier          SubscriptionTier `json:"tier"`
	Name          string           `json:"name"`
	Description   string           `json:"description"`
	MonthlyPrice  float64          `json:"monthly_price"`
	YearlyPrice   float64          `json:"yearly_price"`
	Currency      string           `json:"currency"`
	Features      []string         `json:"features"`
	Limits        *UsageLimits     `json:"limits"`
	OverageRates  map[MetricType]float64 `json:"overage_rates"` // cost per unit over limit
	PopularBadge  bool             `json:"popular_badge"`
	CustomPricing bool             `json:"custom_pricing"`
}

// Credit represents account credits/balance
type Credit struct {
	ID          string    `json:"id"`
	UserID      string    `json:"user_id"`
	Amount      float64   `json:"amount"`
	Currency    string    `json:"currency"`
	Type        string    `json:"type"` // promotional, refund, adjustment
	Reason      string    `json:"reason"`
	ExpiresAt   *time.Time `json:"expires_at,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UsedAmount  float64   `json:"used_amount"`
	Metadata    map[string]string `json:"metadata"`
}

// QuotaEnforcement handles quota checking and enforcement
type QuotaEnforcement struct {
	Enabled      bool                   `json:"enabled"`
	SoftLimit    float64                `json:"soft_limit"`    // warn at this % (e.g., 0.8 for 80%)
	HardLimit    float64                `json:"hard_limit"`    // block at this % (e.g., 1.0 for 100%)
	GracePeriod  time.Duration          `json:"grace_period"`  // time before hard enforcement
	Notifications map[float64]bool      `json:"notifications"` // sent notifications at thresholds
}

// BillingConfig represents billing configuration
type BillingConfig struct {
	Enabled           bool              `json:"enabled"`
	Provider          string            `json:"provider"` // stripe, paypal, crypto
	APIKey            string            `json:"api_key,omitempty"`
	WebhookSecret     string            `json:"webhook_secret,omitempty"`
	TaxRate           float64           `json:"tax_rate"`
	Currency          string            `json:"currency"`
	InvoicePrefix     string            `json:"invoice_prefix"`
	PaymentTerms      int               `json:"payment_terms"` // days
	RetryAttempts     int               `json:"retry_attempts"`
	RetryInterval     time.Duration     `json:"retry_interval"`
	LateFeePercentage float64           `json:"late_fee_percentage"`
	Metadata          map[string]string `json:"metadata"`
}

// MeteringConfig represents metering configuration
type MeteringConfig struct {
	Enabled            bool              `json:"enabled"`
	AggregationInterval time.Duration    `json:"aggregation_interval"`
	RetentionDays      int               `json:"retention_days"`
	BufferSize         int               `json:"buffer_size"`
	FlushInterval      time.Duration     `json:"flush_interval"`
	Metrics            []MetricType      `json:"metrics"`
	Metadata           map[string]string `json:"metadata"`
}
