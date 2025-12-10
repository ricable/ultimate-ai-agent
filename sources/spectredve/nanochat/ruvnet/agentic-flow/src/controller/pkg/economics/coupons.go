package economics

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// CouponManager manages promotional coupons
type CouponManager struct {
	store CouponStore
}

// CouponStore interface for coupon persistence
type CouponStore interface {
	SaveCoupon(ctx context.Context, coupon *Coupon) error
	GetCoupon(ctx context.Context, code string) (*Coupon, error)
	ListCoupons(ctx context.Context, filters map[string]interface{}) ([]*Coupon, error)
	UpdateCoupon(ctx context.Context, coupon *Coupon) error
	IncrementUsage(ctx context.Context, couponID string) error
}

// NewCouponManager creates a new coupon manager
func NewCouponManager(store CouponStore) *CouponManager {
	return &CouponManager{
		store: store,
	}
}

// CreateCoupon creates a new promotional coupon
func (cm *CouponManager) CreateCoupon(ctx context.Context, coupon *Coupon) error {
	// Validate coupon
	if err := cm.validateCoupon(coupon); err != nil {
		return err
	}

	// Normalize code
	coupon.Code = strings.ToUpper(strings.TrimSpace(coupon.Code))

	// Set defaults
	if coupon.Status == "" {
		coupon.Status = "active"
	}

	return cm.store.SaveCoupon(ctx, coupon)
}

// ValidateCoupon validates a coupon code for a subscription
func (cm *CouponManager) ValidateCoupon(ctx context.Context, code string, tier SubscriptionTier, amount float64, userID string, isFirstTime bool) (*CouponValidation, error) {
	// Normalize code
	code = strings.ToUpper(strings.TrimSpace(code))

	// Get coupon
	coupon, err := cm.store.GetCoupon(ctx, code)
	if err != nil {
		return &CouponValidation{
			Valid:   false,
			Message: "Coupon not found",
		}, nil
	}

	// Check status
	if coupon.Status != "active" {
		return &CouponValidation{
			Valid:   false,
			Message: "Coupon is not active",
		}, nil
	}

	// Check validity period
	now := time.Now()
	if now.Before(coupon.ValidFrom) {
		return &CouponValidation{
			Valid:   false,
			Message: "Coupon is not yet valid",
		}, nil
	}
	if now.After(coupon.ValidUntil) {
		return &CouponValidation{
			Valid:   false,
			Message: "Coupon has expired",
		}, nil
	}

	// Check usage limit
	if coupon.MaxUses > 0 && coupon.UsedCount >= coupon.MaxUses {
		return &CouponValidation{
			Valid:   false,
			Message: "Coupon usage limit reached",
		}, nil
	}

	// Check first-time only
	if coupon.FirstTimeOnly && !isFirstTime {
		return &CouponValidation{
			Valid:   false,
			Message: "Coupon is for first-time subscribers only",
		}, nil
	}

	// Check minimum purchase
	if coupon.MinimumPurchase > 0 && amount < coupon.MinimumPurchase {
		return &CouponValidation{
			Valid:   false,
			Message: fmt.Sprintf("Minimum purchase of $%.2f required", coupon.MinimumPurchase),
		}, nil
	}

	// Check applicable tiers
	if len(coupon.ApplicableTiers) > 0 {
		tierApplicable := false
		for _, t := range coupon.ApplicableTiers {
			if t == tier {
				tierApplicable = true
				break
			}
		}
		if !tierApplicable {
			return &CouponValidation{
				Valid:   false,
				Message: "Coupon not applicable to this subscription tier",
			}, nil
		}
	}

	// Calculate discount
	discount := cm.calculateDiscount(coupon, amount)

	return &CouponValidation{
		Valid:          true,
		Message:        "Coupon is valid",
		Coupon:         coupon,
		DiscountAmount: discount,
		FinalAmount:    amount - discount,
	}, nil
}

// ApplyCoupon applies a coupon to a purchase
func (cm *CouponManager) ApplyCoupon(ctx context.Context, couponID string) error {
	return cm.store.IncrementUsage(ctx, couponID)
}

// GeneratePromoCodes generates promotional codes
func (cm *CouponManager) GeneratePromoCodes(ctx context.Context, template *Coupon, count int, prefix string) ([]string, error) {
	codes := make([]string, count)

	for i := 0; i < count; i++ {
		code := fmt.Sprintf("%s-%s", prefix, generateRandomCode(8))
		coupon := *template
		coupon.Code = code
		coupon.ID = generateID()

		if err := cm.store.SaveCoupon(ctx, &coupon); err != nil {
			return nil, err
		}

		codes[i] = code
	}

	return codes, nil
}

// GetActiveCoupons returns all active coupons
func (cm *CouponManager) GetActiveCoupons(ctx context.Context) ([]*Coupon, error) {
	filters := map[string]interface{}{
		"status": "active",
	}
	return cm.store.ListCoupons(ctx, filters)
}

// DeactivateCoupon deactivates a coupon
func (cm *CouponManager) DeactivateCoupon(ctx context.Context, couponID string) error {
	coupon, err := cm.store.GetCoupon(ctx, couponID)
	if err != nil {
		return err
	}

	coupon.Status = "disabled"
	return cm.store.UpdateCoupon(ctx, coupon)
}

// CouponValidation represents coupon validation result
type CouponValidation struct {
	Valid          bool
	Message        string
	Coupon         *Coupon
	DiscountAmount float64
	FinalAmount    float64
}

// Helper methods

func (cm *CouponManager) validateCoupon(coupon *Coupon) error {
	if coupon.Code == "" {
		return fmt.Errorf("coupon code is required")
	}

	if coupon.Type == "" {
		return fmt.Errorf("coupon type is required")
	}

	if coupon.Type != "percentage" && coupon.Type != "fixed" && coupon.Type != "credits" {
		return fmt.Errorf("invalid coupon type: %s", coupon.Type)
	}

	if coupon.Value <= 0 {
		return fmt.Errorf("coupon value must be positive")
	}

	if coupon.Type == "percentage" && coupon.Value > 100 {
		return fmt.Errorf("percentage discount cannot exceed 100%%")
	}

	if coupon.ValidFrom.IsZero() || coupon.ValidUntil.IsZero() {
		return fmt.Errorf("validity period is required")
	}

	if coupon.ValidUntil.Before(coupon.ValidFrom) {
		return fmt.Errorf("invalid validity period")
	}

	return nil
}

func (cm *CouponManager) calculateDiscount(coupon *Coupon, amount float64) float64 {
	switch coupon.Type {
	case "percentage":
		return amount * (coupon.Value / 100)
	case "fixed":
		if coupon.Value > amount {
			return amount
		}
		return coupon.Value
	case "credits":
		// Credits are handled separately
		return 0
	default:
		return 0
	}
}

// Utility functions

func generateRandomCode(length int) string {
	const charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	code := make([]byte, length)
	for i := range code {
		code[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(code)
}

func generateID() string {
	return fmt.Sprintf("cpn_%d", time.Now().UnixNano())
}
