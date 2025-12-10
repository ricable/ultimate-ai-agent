/**
 * Provider Review Interface
 * Handles provider review and approval of patient queries
 */

import {
  PatientQuery,
  QueryReview,
  QueryStatus,
  Provider
} from './types';

export class ReviewInterface {
  private reviews: Map<string, QueryReview[]>;
  private pendingReviews: Map<string, Set<string>>; // providerId -> queryIds

  constructor() {
    this.reviews = new Map();
    this.pendingReviews = new Map();
  }

  /**
   * Assign query for review
   */
  async assignForReview(query: PatientQuery, providerId: string): Promise<void> {
    if (!query.assignedProviderId) {
      query.assignedProviderId = providerId;
    }

    query.status = QueryStatus.IN_REVIEW;
    query.updatedAt = new Date();

    // Add to provider's pending reviews
    const pending = this.pendingReviews.get(providerId) || new Set();
    pending.add(query.id);
    this.pendingReviews.set(providerId, pending);

    console.log(`[REVIEW] Query ${query.id} assigned to provider ${providerId} for review`);
  }

  /**
   * Submit review
   */
  async submitReview(review: QueryReview, query: PatientQuery): Promise<void> {
    // Validate review
    if (!review.queryId || !review.providerId) {
      throw new Error('Invalid review: missing required fields');
    }

    // Update query status based on review action
    switch (review.action) {
      case 'approve':
        query.status = QueryStatus.APPROVED;
        break;
      case 'reject':
        query.status = QueryStatus.REJECTED;
        break;
      case 'escalate':
        query.status = QueryStatus.ESCALATED;
        break;
      case 'request_info':
        query.status = QueryStatus.PENDING;
        break;
    }

    query.reviewedAt = new Date();
    query.updatedAt = new Date();

    // Store review
    const queryReviews = this.reviews.get(query.id) || [];
    queryReviews.push(review);
    this.reviews.set(query.id, queryReviews);

    // Remove from pending reviews
    const pending = this.pendingReviews.get(review.providerId);
    if (pending) {
      pending.delete(query.id);
    }

    console.log(`[REVIEW] Provider ${review.providerId} ${review.action} query ${query.id}`);
  }

  /**
   * Approve query with treatment plan
   */
  async approveQuery(
    queryId: string,
    providerId: string,
    options: {
      diagnosis?: string;
      notes: string;
      recommendations?: string[];
      prescriptions?: any[];
      referrals?: any[];
      followUpRequired: boolean;
      followUpDate?: Date;
    }
  ): Promise<QueryReview> {
    const review: QueryReview = {
      id: `review-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      queryId,
      providerId,
      action: 'approve',
      notes: options.notes,
      diagnosis: options.diagnosis,
      recommendations: options.recommendations,
      prescriptions: options.prescriptions,
      referrals: options.referrals,
      followUpRequired: options.followUpRequired,
      followUpDate: options.followUpDate,
      reviewedAt: new Date()
    };

    return review;
  }

  /**
   * Reject query with reason
   */
  async rejectQuery(
    queryId: string,
    providerId: string,
    reason: string
  ): Promise<QueryReview> {
    const review: QueryReview = {
      id: `review-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      queryId,
      providerId,
      action: 'reject',
      notes: reason,
      followUpRequired: false,
      reviewedAt: new Date()
    };

    return review;
  }

  /**
   * Request additional information
   */
  async requestAdditionalInfo(
    queryId: string,
    providerId: string,
    requestedInfo: string[]
  ): Promise<QueryReview> {
    const review: QueryReview = {
      id: `review-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      queryId,
      providerId,
      action: 'request_info',
      notes: `Additional information requested: ${requestedInfo.join(', ')}`,
      followUpRequired: true,
      reviewedAt: new Date()
    };

    return review;
  }

  /**
   * Escalate query to specialist
   */
  async escalateQuery(
    queryId: string,
    providerId: string,
    reason: string,
    targetSpecialization?: string
  ): Promise<QueryReview> {
    const review: QueryReview = {
      id: `review-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      queryId,
      providerId,
      action: 'escalate',
      notes: reason,
      referrals: targetSpecialization ? [{
        specialization: targetSpecialization,
        reason,
        urgency: 'urgent' as const
      }] : undefined,
      followUpRequired: true,
      reviewedAt: new Date()
    };

    return review;
  }

  /**
   * Get reviews for query
   */
  getReviewsForQuery(queryId: string): QueryReview[] {
    return this.reviews.get(queryId) || [];
  }

  /**
   * Get pending reviews for provider
   */
  getPendingReviewsForProvider(providerId: string): string[] {
    const pending = this.pendingReviews.get(providerId);
    return pending ? Array.from(pending) : [];
  }

  /**
   * Get review history for provider
   */
  getProviderReviewHistory(providerId: string): QueryReview[] {
    const history: QueryReview[] = [];

    for (const reviews of this.reviews.values()) {
      for (const review of reviews) {
        if (review.providerId === providerId) {
          history.push(review);
        }
      }
    }

    return history.sort((a, b) =>
      b.reviewedAt.getTime() - a.reviewedAt.getTime()
    );
  }

  /**
   * Get review statistics
   */
  getReviewStats(providerId: string): {
    totalReviews: number;
    approvalRate: number;
    rejectionRate: number;
    escalationRate: number;
    averageReviewTime: number;
  } {
    const history = this.getProviderReviewHistory(providerId);
    const total = history.length;

    if (total === 0) {
      return {
        totalReviews: 0,
        approvalRate: 0,
        rejectionRate: 0,
        escalationRate: 0,
        averageReviewTime: 0
      };
    }

    let approvals = 0;
    let rejections = 0;
    let escalations = 0;

    for (const review of history) {
      switch (review.action) {
        case 'approve':
          approvals++;
          break;
        case 'reject':
          rejections++;
          break;
        case 'escalate':
          escalations++;
          break;
      }
    }

    return {
      totalReviews: total,
      approvalRate: (approvals / total) * 100,
      rejectionRate: (rejections / total) * 100,
      escalationRate: (escalations / total) * 100,
      averageReviewTime: 0 // Would calculate from query timestamps
    };
  }
}
