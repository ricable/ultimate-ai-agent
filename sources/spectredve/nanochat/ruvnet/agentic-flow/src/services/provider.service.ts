/**
 * Provider Service
 * Manages healthcare provider interactions and notifications
 */

import { v4 as uuidv4 } from 'uuid';
import type {
  Provider,
  ProviderReview,
  AnalysisResult,
  NotificationPreferences
} from '../types/medical.types';

interface PendingReview {
  analysisId: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  timestamp: Date;
  notified: boolean;
}

export class ProviderService {
  private providers: Map<string, Provider> = new Map();
  private pendingReviews: Map<string, PendingReview> = new Map();
  private reviews: Map<string, ProviderReview> = new Map();

  constructor() {
    // Initialize with default provider
    this.addProvider({
      id: 'default-provider',
      name: 'Dr. Default',
      specialty: 'General Medicine',
      credentials: ['MD', 'FACP'],
      licenseNumber: 'MED-12345',
      email: 'provider@example.com',
      phone: '+1-555-0100',
      notificationPreferences: {
        email: true,
        sms: true,
        push: true,
        urgentOnly: false,
        minimumSeverity: 'medium'
      }
    });
  }

  /**
   * Add a healthcare provider
   */
  public addProvider(provider: Provider): void {
    this.providers.set(provider.id, provider);
  }

  /**
   * Get provider by ID
   */
  public getProvider(id: string): Provider | null {
    return this.providers.get(id) || null;
  }

  /**
   * Notify provider about analysis requiring review
   */
  public async notifyProvider(
    analysisId: string,
    analysis: AnalysisResult,
    urgent: boolean = false
  ): Promise<void> {
    try {
      // Determine priority
      const priority = this.determinePriority(analysis, urgent);

      // Get default provider (in production, use assignment logic)
      const provider = Array.from(this.providers.values())[0];
      if (!provider) {
        throw new Error('No provider available');
      }

      // Check notification preferences
      if (this.shouldNotify(provider.notificationPreferences, priority)) {
        // Send notifications
        await this.sendNotifications(provider, analysisId, analysis, priority);

        // Add to pending reviews
        this.pendingReviews.set(analysisId, {
          analysisId,
          priority,
          timestamp: new Date(),
          notified: true
        });

        console.log(`Provider ${provider.name} notified about analysis ${analysisId}`);
      }
    } catch (error) {
      console.error('Error notifying provider:', error);
      throw error;
    }
  }

  /**
   * Submit provider review
   */
  public async submitReview(
    analysisId: string,
    reviewData: {
      decision: 'approved' | 'rejected' | 'modified';
      comments?: string;
      modifications?: string[];
    }
  ): Promise<void> {
    try {
      // Get provider (in production, from authentication)
      const provider = Array.from(this.providers.values())[0];
      if (!provider) {
        throw new Error('No provider found');
      }

      const review: ProviderReview = {
        providerId: provider.id,
        providerName: provider.name,
        timestamp: new Date(),
        decision: reviewData.decision,
        comments: reviewData.comments,
        modifications: reviewData.modifications
      };

      this.reviews.set(analysisId, review);

      // Remove from pending reviews
      this.pendingReviews.delete(analysisId);

      console.log(`Provider review submitted for analysis ${analysisId}: ${reviewData.decision}`);
    } catch (error) {
      console.error('Error submitting review:', error);
      throw error;
    }
  }

  /**
   * Get pending reviews
   */
  public async getPendingReviews(): Promise<PendingReview[]> {
    return Array.from(this.pendingReviews.values());
  }

  /**
   * Get review for analysis
   */
  public async getReview(analysisId: string): Promise<ProviderReview | null> {
    return this.reviews.get(analysisId) || null;
  }

  /**
   * Determine notification priority
   */
  private determinePriority(
    analysis: AnalysisResult,
    urgent: boolean
  ): 'low' | 'medium' | 'high' | 'urgent' {
    if (urgent) return 'urgent';

    // Check for emergency warnings
    const hasEmergency = analysis.warnings.some(w => w.type === 'emergency');
    if (hasEmergency) return 'urgent';

    // Check confidence score
    if (analysis.confidenceScore.overall < 0.50) return 'high';
    if (analysis.confidenceScore.overall < 0.70) return 'medium';

    // Check for hallucination warnings
    const hasHallucination = analysis.warnings.some(w => w.type === 'hallucination');
    if (hasHallucination) return 'high';

    return 'medium';
  }

  /**
   * Check if provider should be notified based on preferences
   */
  private shouldNotify(
    prefs: NotificationPreferences,
    priority: 'low' | 'medium' | 'high' | 'urgent'
  ): boolean {
    if (prefs.urgentOnly && priority !== 'urgent') {
      return false;
    }

    const severityLevels = ['low', 'medium', 'high', 'urgent'];
    const minIndex = severityLevels.indexOf(prefs.minimumSeverity);
    const currentIndex = severityLevels.indexOf(priority);

    return currentIndex >= minIndex;
  }

  /**
   * Send notifications via configured channels
   */
  private async sendNotifications(
    provider: Provider,
    analysisId: string,
    analysis: AnalysisResult,
    priority: 'low' | 'medium' | 'high' | 'urgent'
  ): Promise<void> {
    const prefs = provider.notificationPreferences;

    // Email notification
    if (prefs.email) {
      await this.sendEmail(provider.email, analysisId, analysis, priority);
    }

    // SMS notification
    if (prefs.sms && provider.phone) {
      await this.sendSMS(provider.phone, analysisId, analysis, priority);
    }

    // Push notification
    if (prefs.push) {
      await this.sendPushNotification(provider.id, analysisId, analysis, priority);
    }
  }

  /**
   * Send email notification
   */
  private async sendEmail(
    email: string,
    analysisId: string,
    analysis: AnalysisResult,
    priority: string
  ): Promise<void> {
    // In production, integrate with email service (SendGrid, AWS SES, etc.)
    console.log(`📧 Email sent to ${email}:`);
    console.log(`   Priority: ${priority.toUpperCase()}`);
    console.log(`   Analysis ID: ${analysisId}`);
    console.log(`   Confidence: ${(analysis.confidenceScore.overall * 100).toFixed(1)}%`);
    console.log(`   Requires Review: ${analysis.requiresProviderReview ? 'Yes' : 'No'}`);
  }

  /**
   * Send SMS notification
   */
  private async sendSMS(
    phone: string,
    analysisId: string,
    analysis: AnalysisResult,
    priority: string
  ): Promise<void> {
    // In production, integrate with SMS service (Twilio, AWS SNS, etc.)
    console.log(`📱 SMS sent to ${phone}:`);
    console.log(`   [${priority.toUpperCase()}] Analysis ${analysisId.substring(0, 8)} requires review`);
  }

  /**
   * Send push notification
   */
  private async sendPushNotification(
    providerId: string,
    analysisId: string,
    analysis: AnalysisResult,
    priority: string
  ): Promise<void> {
    // In production, integrate with push service (Firebase, OneSignal, etc.)
    console.log(`🔔 Push notification sent to provider ${providerId}:`);
    console.log(`   Priority: ${priority.toUpperCase()}`);
    console.log(`   Analysis: ${analysisId}`);
  }

  /**
   * List all providers
   */
  public listProviders(): Provider[] {
    return Array.from(this.providers.values());
  }
}
