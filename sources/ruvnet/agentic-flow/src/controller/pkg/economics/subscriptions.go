package economics

import (
	"context"
	"fmt"
	"time"
)

// SubscriptionManager manages subscriptions
type SubscriptionManager struct {
	pricingManager  *PricingManager
	meteringEngine  *MeteringEngine
	paymentProcessor *PaymentProcessor
	couponManager   *CouponManager
	store           SubscriptionStore
}

// SubscriptionStore interface for subscription persistence
type SubscriptionStore interface {
	SaveSubscription(ctx context.Context, sub *Subscription) error
	GetSubscription(ctx context.Context, id string) (*Subscription, error)
	GetUserSubscription(ctx context.Context, userID string) (*Subscription, error)
	UpdateSubscription(ctx context.Context, sub *Subscription) error
	ListSubscriptions(ctx context.Context, filters map[string]interface{}) ([]*Subscription, error)
}

// NewSubscriptionManager creates a new subscription manager
func NewSubscriptionManager(
	pricingManager *PricingManager,
	meteringEngine *MeteringEngine,
	paymentProcessor *PaymentProcessor,
	couponManager *CouponManager,
	store SubscriptionStore,
) *SubscriptionManager {
	return &SubscriptionManager{
		pricingManager:  pricingManager,
		meteringEngine:  meteringEngine,
		paymentProcessor: paymentProcessor,
		couponManager:   couponManager,
		store:           store,
	}
}

// CreateSubscription creates a new subscription
func (sm *SubscriptionManager) CreateSubscription(ctx context.Context, req *CreateSubscriptionRequest) (*Subscription, error) {
	// Get pricing tier
	tier, err := sm.pricingManager.GetTier(req.Tier)
	if err != nil {
		return nil, err
	}

	// Calculate price
	price, err := sm.pricingManager.CalculatePrice(req.Tier, req.BillingCycle, nil)
	if err != nil {
		return nil, err
	}

	// Apply coupon if provided
	if req.CouponCode != "" {
		validation, err := sm.couponManager.ValidateCoupon(ctx, req.CouponCode, req.Tier, price, req.UserID, req.IsFirstTime)
		if err != nil {
			return nil, err
		}
		if !validation.Valid {
			return nil, fmt.Errorf("invalid coupon: %s", validation.Message)
		}
		price = validation.FinalAmount
	}

	// Process payment
	if req.PaymentMethodID != "" && price > 0 {
		payment, err := sm.paymentProcessor.ChargePaymentMethod(ctx, req.PaymentMethodID, price, "USD")
		if err != nil {
			return nil, fmt.Errorf("payment failed: %w", err)
		}
		if payment.Status != "succeeded" {
			return nil, fmt.Errorf("payment failed: %s", payment.FailureReason)
		}
	}

	// Create subscription
	now := time.Now()
	sub := &Subscription{
		ID:              generateSubscriptionID(),
		UserID:          req.UserID,
		Tier:            req.Tier,
		BillingCycle:    req.BillingCycle,
		Status:          "active",
		StartDate:       now,
		EndDate:         sm.calculateEndDate(now, req.BillingCycle),
		RenewalDate:     sm.calculateRenewalDate(now, req.BillingCycle),
		Price:           price,
		Currency:        "USD",
		Features:        tier.Features,
		Limits:          tier.Limits,
		PaymentMethodID: req.PaymentMethodID,
		Metadata:        req.Metadata,
	}

	if err := sm.store.SaveSubscription(ctx, sub); err != nil {
		return nil, err
	}

	return sub, nil
}

// UpgradeSubscription upgrades a subscription to a higher tier
func (sm *SubscriptionManager) UpgradeSubscription(ctx context.Context, subID string, newTier SubscriptionTier) (*Subscription, error) {
	// Get current subscription
	sub, err := sm.store.GetSubscription(ctx, subID)
	if err != nil {
		return nil, err
	}

	if sub.Status != "active" {
		return nil, fmt.Errorf("subscription is not active")
	}

	// Get new pricing
	newPricing, err := sm.pricingManager.GetTier(newTier)
	if err != nil {
		return nil, err
	}

	// Calculate prorated amount
	proratedAmount := sm.calculateProratedAmount(sub, newPricing)

	// Process prorated payment
	if proratedAmount > 0 && sub.PaymentMethodID != "" {
		payment, err := sm.paymentProcessor.ChargePaymentMethod(ctx, sub.PaymentMethodID, proratedAmount, sub.Currency)
		if err != nil {
			return nil, fmt.Errorf("payment failed: %w", err)
		}
		if payment.Status != "succeeded" {
			return nil, fmt.Errorf("payment failed: %s", payment.FailureReason)
		}
	}

	// Update subscription
	sub.Tier = newTier
	sub.Features = newPricing.Features
	sub.Limits = newPricing.Limits
	sub.Price = sm.calculateNewPrice(newTier, sub.BillingCycle)

	if err := sm.store.UpdateSubscription(ctx, sub); err != nil {
		return nil, err
	}

	return sub, nil
}

// CancelSubscription cancels a subscription
func (sm *SubscriptionManager) CancelSubscription(ctx context.Context, subID string, immediate bool) error {
	sub, err := sm.store.GetSubscription(ctx, subID)
	if err != nil {
		return err
	}

	if immediate {
		sub.Status = "cancelled"
		sub.EndDate = time.Now()
	} else {
		// Cancel at end of billing period
		sub.Status = "cancelling"
	}

	return sm.store.UpdateSubscription(ctx, sub)
}

// RenewSubscription renews a subscription
func (sm *SubscriptionManager) RenewSubscription(ctx context.Context, subID string) error {
	sub, err := sm.store.GetSubscription(ctx, subID)
	if err != nil {
		return err
	}

	// Calculate renewal price
	price, err := sm.pricingManager.CalculatePrice(sub.Tier, sub.BillingCycle, nil)
	if err != nil {
		return err
	}

	// Process payment
	if sub.PaymentMethodID != "" && price > 0 {
		payment, err := sm.paymentProcessor.ChargePaymentMethod(ctx, sub.PaymentMethodID, price, sub.Currency)
		if err != nil {
			return fmt.Errorf("renewal payment failed: %w", err)
		}
		if payment.Status != "succeeded" {
			sub.Status = "past_due"
			sm.store.UpdateSubscription(ctx, sub)
			return fmt.Errorf("renewal payment failed: %s", payment.FailureReason)
		}
	}

	// Update subscription
	now := time.Now()
	sub.StartDate = now
	sub.EndDate = sm.calculateEndDate(now, sub.BillingCycle)
	sub.RenewalDate = sm.calculateRenewalDate(now, sub.BillingCycle)
	sub.Status = "active"

	return sm.store.UpdateSubscription(ctx, sub)
}

// CheckAccessLimit checks if user can perform an action based on their subscription
func (sm *SubscriptionManager) CheckAccessLimit(ctx context.Context, userID string, metric MetricType, amount float64) (*AccessCheck, error) {
	// Get user subscription
	sub, err := sm.store.GetUserSubscription(ctx, userID)
	if err != nil {
		return nil, err
	}

	if sub.Status != "active" {
		return &AccessCheck{
			Allowed: false,
			Reason:  "Subscription is not active",
		}, nil
	}

	// Check quota
	quotaCheck, err := sm.meteringEngine.CheckQuota(ctx, userID, metric, amount, sub.Limits)
	if err != nil {
		return nil, err
	}

	if !quotaCheck.Allowed {
		return &AccessCheck{
			Allowed:       false,
			Reason:        "Quota exceeded",
			CurrentUsage:  quotaCheck.Current,
			Limit:         quotaCheck.Limit,
			PercentUsed:   quotaCheck.PercentUsed,
			UpgradeOption: sm.suggestUpgrade(sub.Tier),
		}, nil
	}

	return &AccessCheck{
		Allowed:      true,
		CurrentUsage: quotaCheck.Current,
		Limit:        quotaCheck.Limit,
		PercentUsed:  quotaCheck.PercentUsed,
	}, nil
}

// CreateSubscriptionRequest represents a subscription creation request
type CreateSubscriptionRequest struct {
	UserID          string
	Tier            SubscriptionTier
	BillingCycle    BillingCycle
	PaymentMethodID string
	CouponCode      string
	IsFirstTime     bool
	Metadata        map[string]string
}

// AccessCheck represents an access check result
type AccessCheck struct {
	Allowed       bool
	Reason        string
	CurrentUsage  float64
	Limit         float64
	PercentUsed   float64
	UpgradeOption *UpgradeOption
}

// UpgradeOption represents an upgrade suggestion
type UpgradeOption struct {
	Tier        SubscriptionTier
	Name        string
	Price       float64
	Benefits    []string
	LimitIncrease string
}

// Helper methods

func (sm *SubscriptionManager) calculateEndDate(start time.Time, cycle BillingCycle) time.Time {
	switch cycle {
	case CycleMonthly:
		return start.AddDate(0, 1, 0)
	case CycleYearly:
		return start.AddDate(1, 0, 0)
	case CycleQuarterly:
		return start.AddDate(0, 3, 0)
	default:
		return start.AddDate(0, 1, 0)
	}
}

func (sm *SubscriptionManager) calculateRenewalDate(start time.Time, cycle BillingCycle) time.Time {
	end := sm.calculateEndDate(start, cycle)
	return end.AddDate(0, 0, -7) // 7 days before end
}

func (sm *SubscriptionManager) calculateProratedAmount(sub *Subscription, newPricing *PricingTier) float64 {
	// Calculate prorated amount based on remaining time
	now := time.Now()
	total := sub.EndDate.Sub(sub.StartDate)
	remaining := sub.EndDate.Sub(now)

	if remaining <= 0 || total <= 0 {
		return 0
	}

	prorateRatio := float64(remaining) / float64(total)

	var newPrice float64
	switch sub.BillingCycle {
	case CycleMonthly:
		newPrice = newPricing.MonthlyPrice
	case CycleYearly:
		newPrice = newPricing.YearlyPrice
	default:
		newPrice = newPricing.MonthlyPrice
	}

	return (newPrice - sub.Price) * prorateRatio
}

func (sm *SubscriptionManager) calculateNewPrice(tier SubscriptionTier, cycle BillingCycle) float64 {
	pricingTier, _ := sm.pricingManager.GetTier(tier)
	if pricingTier == nil {
		return 0
	}

	switch cycle {
	case CycleMonthly:
		return pricingTier.MonthlyPrice
	case CycleYearly:
		return pricingTier.YearlyPrice
	case CycleQuarterly:
		return pricingTier.YearlyPrice / 4
	default:
		return pricingTier.MonthlyPrice
	}
}

func (sm *SubscriptionManager) suggestUpgrade(currentTier SubscriptionTier) *UpgradeOption {
	var nextTier SubscriptionTier

	switch currentTier {
	case TierFree:
		nextTier = TierStarter
	case TierStarter:
		nextTier = TierPro
	case TierPro:
		nextTier = TierEnterprise
	default:
		return nil
	}

	pricing, _ := sm.pricingManager.GetTier(nextTier)
	if pricing == nil {
		return nil
	}

	return &UpgradeOption{
		Tier:     nextTier,
		Name:     pricing.Name,
		Price:    pricing.MonthlyPrice,
		Benefits: pricing.Features,
	}
}

func generateSubscriptionID() string {
	return fmt.Sprintf("sub_%d", time.Now().UnixNano())
}
