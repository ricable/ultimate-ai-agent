/**
 * Type definitions for healthcare notification system
 */

export enum NotificationChannel {
  EMAIL = 'email',
  SMS = 'sms',
  WEBHOOK = 'webhook',
  WEBSOCKET = 'websocket',
  INAPP = 'inapp'
}

export enum NotificationPriority {
  EMERGENCY = 'emergency',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low'
}

export enum NotificationStatus {
  PENDING = 'pending',
  SENT = 'sent',
  DELIVERED = 'delivered',
  FAILED = 'failed',
  READ = 'read'
}

export interface NotificationRecipient {
  id: string;
  type: 'provider' | 'patient';
  name: string;
  email?: string;
  phone?: string;
  preferredChannels: NotificationChannel[];
}

export interface NotificationPayload {
  id: string;
  type: string;
  title: string;
  message: string;
  priority: NotificationPriority;
  channels: NotificationChannel[];
  recipient: NotificationRecipient;
  metadata?: Record<string, any>;
  createdAt: Date;
  expiresAt?: Date;
  requiresConsent?: boolean;
  hipaaCompliant: boolean;
}

export interface NotificationResult {
  id: string;
  channel: NotificationChannel;
  status: NotificationStatus;
  sentAt?: Date;
  deliveredAt?: Date;
  error?: string;
  metadata?: Record<string, any>;
}

export interface NotificationConfig {
  channels: {
    email?: EmailConfig;
    sms?: SMSConfig;
    webhook?: WebhookConfig;
    websocket?: WebSocketConfig;
    inapp?: InAppConfig;
  };
  retryPolicy: {
    maxAttempts: number;
    backoffMs: number;
    exponentialBackoff: boolean;
  };
  security: {
    encryptPayload: boolean;
    requireSignature: boolean;
    auditLog: boolean;
  };
}

export interface EmailConfig {
  host: string;
  port: number;
  secure: boolean;
  auth: {
    user: string;
    pass: string;
  };
  from: string;
  replyTo?: string;
}

export interface SMSConfig {
  provider: 'twilio' | 'aws-sns' | 'vonage';
  accountSid: string;
  authToken: string;
  fromNumber: string;
  maxLength?: number;
}

export interface WebhookConfig {
  endpoints: Map<string, string>;
  timeout: number;
  headers?: Record<string, string>;
  retryOnFailure: boolean;
}

export interface WebSocketConfig {
  port: number;
  path: string;
  secure: boolean;
  heartbeatInterval: number;
}

export interface InAppConfig {
  storageType: 'memory' | 'database';
  maxNotificationsPerUser: number;
  retentionDays: number;
}

export interface INotifier {
  send(payload: NotificationPayload): Promise<NotificationResult>;
  getStatus(notificationId: string): Promise<NotificationStatus>;
  cancel(notificationId: string): Promise<boolean>;
}
