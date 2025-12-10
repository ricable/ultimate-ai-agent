/**
 * Payment Processing
 * Integration with agentic-payments library
 */

import {
  PaymentProvider,
  PaymentRequest,
  PaymentResult,
  PaymentStatus,
  PaymentMethod,
  StorageAdapter
} from '../types.js';

export interface PaymentConfig {
  provider: PaymentProvider;
  apiKey?: string;
  secretKey?: string;
  webhookSecret?: string;
  testMode?: boolean;
}

export class PaymentProcessor {
  private config: PaymentConfig;
  private storage?: StorageAdapter;

  constructor(config: PaymentConfig, storage?: StorageAdapter) {
    this.config = config;
    this.storage = storage;
  }

  /**
   * Process a payment
   */
  async processPayment(request: PaymentRequest): Promise<PaymentResult> {
    try {
      const result = await this.chargePaymentMethod(
        request.paymentMethodId,
        request.amount,
        request.currency,
        request.description,
        request.metadata
      );

      return result;
    } catch (error) {
      return {
        transactionId: this.generateTransactionId(),
        status: PaymentStatus.Failed,
        amount: request.amount,
        currency: request.currency,
        error: error instanceof Error ? error.message : 'Payment failed',
        processedAt: new Date()
      };
    }
  }

  /**
   * Charge a payment method
   */
  async chargePaymentMethod(
    paymentMethodId: string,
    amount: number,
    currency: string,
    description?: string,
    metadata?: Record<string, any>
  ): Promise<PaymentResult> {
    // In production, this would integrate with agentic-payments
    // For now, simulate payment processing

    const result: PaymentResult = {
      transactionId: this.generateTransactionId(),
      status: PaymentStatus.Succeeded,
      amount,
      currency,
      processedAt: new Date()
    };

    // Simulate provider-specific transaction IDs
    switch (this.config.provider) {
      case PaymentProvider.Stripe:
        result.providerTransactionId = `pi_${this.generateRandomCode(24)}`;
        break;

      case PaymentProvider.PayPal:
        result.providerTransactionId = `PAYID-${this.generateRandomCode(20)}`;
        break;

      case PaymentProvider.Crypto:
        result.providerTransactionId = `0x${this.generateRandomCode(64)}`;
        break;
    }

    // Simulate 1% payment failure rate in test mode
    if (this.config.testMode && Math.random() < 0.01) {
      result.status = PaymentStatus.Failed;
      result.error = 'Card declined';
    }

    return result;
  }

  /**
   * Refund a payment
   */
  async refundPayment(
    transactionId: string,
    amount?: number,
    reason?: string
  ): Promise<PaymentResult> {
    // Simulate refund processing
    return {
      transactionId: this.generateTransactionId(),
      status: PaymentStatus.Refunded,
      amount: amount || 0,
      currency: 'USD',
      providerTransactionId: `re_${this.generateRandomCode(24)}`,
      processedAt: new Date()
    };
  }

  /**
   * Validate payment method
   */
  async validatePaymentMethod(paymentMethodId: string): Promise<boolean> {
    // In production, validate with payment provider
    // For now, accept any non-empty ID
    return paymentMethodId.length > 0;
  }

  /**
   * Create setup intent (for adding payment methods)
   */
  async createSetupIntent(userId: string): Promise<{
    clientSecret: string;
    setupIntentId: string;
  }> {
    return {
      clientSecret: `seti_${this.generateRandomCode(32)}`,
      setupIntentId: `seti_${this.generateRandomCode(24)}`
    };
  }

  /**
   * Create payment intent
   */
  async createPaymentIntent(
    amount: number,
    currency: string,
    metadata?: Record<string, any>
  ): Promise<{
    clientSecret: string;
    paymentIntentId: string;
  }> {
    return {
      clientSecret: `pi_${this.generateRandomCode(32)}`,
      paymentIntentId: `pi_${this.generateRandomCode(24)}`
    };
  }

  /**
   * Handle webhook events from payment provider
   */
  async handleWebhook(
    payload: string,
    signature: string
  ): Promise<{
    event: string;
    data: any;
  }> {
    // Verify webhook signature
    if (this.config.webhookSecret) {
      const isValid = this.verifyWebhookSignature(payload, signature);
      if (!isValid) {
        throw new Error('Invalid webhook signature');
      }
    }

    // Parse webhook payload
    const event = JSON.parse(payload);
    return event;
  }

  private verifyWebhookSignature(payload: string, signature: string): boolean {
    // In production, verify with provider's signature algorithm
    // For now, simple check
    return signature.length > 0;
  }

  private generateTransactionId(): string {
    return `txn_${Date.now()}_${this.generateRandomCode(16)}`;
  }

  private generateRandomCode(length: number): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  }
}

/**
 * Payment Provider Factory
 * Creates payment processor for specific provider
 */
export class PaymentProviderFactory {
  static create(provider: PaymentProvider, config: any, storage?: StorageAdapter): PaymentProcessor {
    const paymentConfig: PaymentConfig = {
      provider,
      ...config
    };

    return new PaymentProcessor(paymentConfig, storage);
  }

  static createStripe(apiKey: string, storage?: StorageAdapter): PaymentProcessor {
    return this.create(PaymentProvider.Stripe, { apiKey }, storage);
  }

  static createPayPal(clientId: string, clientSecret: string, storage?: StorageAdapter): PaymentProcessor {
    return this.create(PaymentProvider.PayPal, { apiKey: clientId, secretKey: clientSecret }, storage);
  }

  static createCrypto(walletAddress: string, storage?: StorageAdapter): PaymentProcessor {
    return this.create(PaymentProvider.Crypto, { apiKey: walletAddress }, storage);
  }
}
