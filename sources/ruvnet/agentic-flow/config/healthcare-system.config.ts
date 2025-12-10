/**
 * Healthcare Provider System Configuration
 * Complete configuration for notifications, providers, consent, routing, and security
 */

import {
  NotificationConfig,
  EmailConfig,
  SMSConfig,
  WebhookConfig,
  WebSocketConfig,
  InAppConfig
} from '../src/notifications/types';
import { QueueConfig } from '../src/providers/types';

/**
 * Email Configuration (SMTP)
 */
export const emailConfig: EmailConfig = {
  host: process.env.SMTP_HOST || 'smtp.example.com',
  port: parseInt(process.env.SMTP_PORT || '587'),
  secure: process.env.SMTP_SECURE === 'true',
  auth: {
    user: process.env.SMTP_USER || 'notifications@healthcare.example.com',
    pass: process.env.SMTP_PASSWORD || ''
  },
  from: process.env.SMTP_FROM || 'Healthcare System <noreply@healthcare.example.com>',
  replyTo: process.env.SMTP_REPLY_TO || 'support@healthcare.example.com'
};

/**
 * SMS Configuration (Twilio)
 */
export const smsConfig: SMSConfig = {
  provider: 'twilio',
  accountSid: process.env.TWILIO_ACCOUNT_SID || '',
  authToken: process.env.TWILIO_AUTH_TOKEN || '',
  fromNumber: process.env.TWILIO_FROM_NUMBER || '+1234567890',
  maxLength: 160
};

/**
 * Webhook Configuration
 */
export const webhookConfig: WebhookConfig = {
  endpoints: new Map([
    // Add provider webhook endpoints here
    // ['provider-id-1', 'https://provider1.example.com/webhook'],
  ]),
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
    'X-Healthcare-System': 'v1'
  },
  retryOnFailure: true
};

/**
 * WebSocket Configuration
 */
export const websocketConfig: WebSocketConfig = {
  port: parseInt(process.env.WS_PORT || '8080'),
  path: '/notifications',
  secure: process.env.WS_SECURE === 'true',
  heartbeatInterval: 30000 // 30 seconds
};

/**
 * In-App Notification Configuration
 */
export const inappConfig: InAppConfig = {
  storageType: 'memory', // In production, use 'database'
  maxNotificationsPerUser: 100,
  retentionDays: 30
};

/**
 * Complete Notification System Configuration
 */
export const notificationConfig: NotificationConfig = {
  channels: {
    email: emailConfig,
    sms: smsConfig,
    webhook: webhookConfig,
    websocket: websocketConfig,
    inapp: inappConfig
  },
  retryPolicy: {
    maxAttempts: 3,
    backoffMs: 1000,
    exponentialBackoff: true
  },
  security: {
    encryptPayload: true,
    requireSignature: true,
    auditLog: true
  }
};

/**
 * Patient Queue Configuration
 */
export const queueConfig: QueueConfig = {
  autoAssignment: true,
  priorityWeighting: new Map([
    ['emergency', 4],
    ['urgent', 3],
    ['routine', 2],
    ['low', 1]
  ]),
  maxQueueSize: 1000,
  stalePeriodMinutes: 30
};

/**
 * HIPAA Security Configuration
 */
export const hipaaSecurityConfig = {
  encryptionKey: process.env.ENCRYPTION_KEY || 'change-this-in-production',
  auditEnabled: true,
  minPasswordLength: 12,
  sessionTimeoutMinutes: 30,
  maxLoginAttempts: 3,
  requireMFA: true
};

/**
 * Provider System Configuration
 */
export const providerConfig = {
  maxConcurrentCasesDefault: 10,
  responseTimeoutMinutes: 15,
  escalationTimeoutMinutes: 30,
  performanceTrackingEnabled: true
};

/**
 * Routing Configuration
 */
export const routingConfig = {
  emergencyResponseTimeMinutes: 5,
  urgentResponseTimeMinutes: 15,
  routineResponseTimeMinutes: 60,
  autoEscalationEnabled: true,
  providerMatchMinScore: 0.5
};

/**
 * Consent Management Configuration
 */
export const consentConfig = {
  defaultExpiryDays: 365,
  renewalReminderDays: 30,
  requireWitnessSignature: false,
  digitalSignatureEnabled: true
};

/**
 * System-wide Configuration
 */
export const systemConfig = {
  environment: process.env.NODE_ENV || 'development',
  apiVersion: 'v1',
  logLevel: process.env.LOG_LEVEL || 'info',
  enableMetrics: true,
  enableHealthChecks: true
};

/**
 * Export complete configuration
 */
export default {
  notifications: notificationConfig,
  queue: queueConfig,
  hipaa: hipaaSecurityConfig,
  providers: providerConfig,
  routing: routingConfig,
  consent: consentConfig,
  system: systemConfig
};
