/**
 * Coupon Management System
 * Promotional codes and discounts
 */

import {
  Coupon,
  CouponType,
  CouponValidation,
  SubscriptionTier,
  StorageAdapter
} from '../types.js';

export class CouponManager {
  private storage: StorageAdapter;

  constructor(storage: StorageAdapter) {
    this.storage = storage;
  }

  /**
   * Create a new coupon
   */
  async createCoupon(params: {
    code: string;
    type: CouponType;
    value: number;
    description?: string;
    maxRedemptions?: number;
    validFrom?: Date;
    validUntil?: Date;
    applicableTiers?: SubscriptionTier[];
    minimumAmount?: number;
    currency?: string;
  }): Promise<Coupon> {
    const {
      code,
      type,
      value,
      description,
      maxRedemptions,
      validFrom,
      validUntil,
      applicableTiers,
      minimumAmount,
      currency
    } = params;

    // Check if code already exists
    const existing = await this.storage.getCoupon(code);
    if (existing) {
      throw new Error(`Coupon code already exists: ${code}`);
    }

    const coupon: Coupon = {
      id: this.generateId(),
      code: code.toUpperCase(),
      type,
      value,
      currency: type === CouponType.Fixed ? (currency || 'USD') : undefined,
      description,
      maxRedemptions,
      timesRedeemed: 0,
      validFrom: validFrom || new Date(),
      validUntil,
      applicableTiers,
      minimumAmount,
      active: true,
      createdAt: new Date()
    };

    await this.storage.saveCoupon(coupon);
    return coupon;
  }

  /**
   * Validate a coupon
   */
  async validateCoupon(
    code: string,
    tier: SubscriptionTier,
    amount: number
  ): Promise<CouponValidation> {
    const coupon = await this.storage.getCoupon(code.toUpperCase());

    if (!coupon) {
      return {
        valid: false,
        discountAmount: 0,
        finalAmount: amount,
        error: 'Invalid coupon code'
      };
    }

    // Check if active
    if (!coupon.active) {
      return {
        valid: false,
        discountAmount: 0,
        finalAmount: amount,
        error: 'Coupon is no longer active'
      };
    }

    // Check validity dates
    const now = new Date();
    if (now < coupon.validFrom) {
      return {
        valid: false,
        discountAmount: 0,
        finalAmount: amount,
        error: 'Coupon is not yet valid'
      };
    }

    if (coupon.validUntil && now > coupon.validUntil) {
      return {
        valid: false,
        discountAmount: 0,
        finalAmount: amount,
        error: 'Coupon has expired'
      };
    }

    // Check redemption limit
    if (coupon.maxRedemptions && coupon.timesRedeemed >= coupon.maxRedemptions) {
      return {
        valid: false,
        discountAmount: 0,
        finalAmount: amount,
        error: 'Coupon has reached maximum redemptions'
      };
    }

    // Check tier applicability
    if (coupon.applicableTiers && !coupon.applicableTiers.includes(tier)) {
      return {
        valid: false,
        discountAmount: 0,
        finalAmount: amount,
        error: `Coupon not applicable to ${tier} tier`
      };
    }

    // Check minimum amount
    if (coupon.minimumAmount && amount < coupon.minimumAmount) {
      return {
        valid: false,
        discountAmount: 0,
        finalAmount: amount,
        error: `Minimum amount of ${coupon.minimumAmount} required`
      };
    }

    // Calculate discount
    const discountAmount = this.calculateDiscount(coupon, amount);
    const finalAmount = Math.max(0, amount - discountAmount);

    return {
      valid: true,
      coupon,
      discountAmount,
      finalAmount
    };
  }

  /**
   * Apply (redeem) a coupon
   */
  async applyCoupon(code: string): Promise<Coupon> {
    const coupon = await this.storage.getCoupon(code.toUpperCase());

    if (!coupon) {
      throw new Error('Coupon not found');
    }

    coupon.timesRedeemed++;

    // Deactivate if max redemptions reached
    if (coupon.maxRedemptions && coupon.timesRedeemed >= coupon.maxRedemptions) {
      coupon.active = false;
    }

    await this.storage.updateCoupon(coupon);
    return coupon;
  }

  /**
   * Deactivate a coupon
   */
  async deactivateCoupon(code: string): Promise<Coupon> {
    const coupon = await this.storage.getCoupon(code.toUpperCase());

    if (!coupon) {
      throw new Error('Coupon not found');
    }

    coupon.active = false;
    await this.storage.updateCoupon(coupon);
    return coupon;
  }

  /**
   * List all coupons
   */
  async listCoupons(activeOnly: boolean = false): Promise<Coupon[]> {
    const coupons = await this.storage.listCoupons();

    if (activeOnly) {
      return coupons.filter(c => c.active);
    }

    return coupons;
  }

  private calculateDiscount(coupon: Coupon, amount: number): number {
    switch (coupon.type) {
      case CouponType.Percentage:
        return (amount * coupon.value) / 100;

      case CouponType.Fixed:
        return coupon.value;

      case CouponType.Credit:
        return Math.min(coupon.value, amount);

      default:
        return 0;
    }
  }

  private generateId(): string {
    return `coupon_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}
