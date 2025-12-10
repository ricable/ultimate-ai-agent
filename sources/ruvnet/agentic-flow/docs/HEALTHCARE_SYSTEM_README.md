# Healthcare Provider Notification and Communication System

## Overview

A comprehensive, HIPAA-compliant healthcare provider notification and communication system with multi-channel notifications, patient queue management, consent tracking, and intelligent routing.

## Features

### 1. Multi-Channel Notifications (`src/notifications/`)

- **Email Notifications**: SMTP-based with HTML templates
- **SMS Notifications**: Twilio/AWS SNS integration
- **Webhook Notifications**: HTTP/HTTPS with retry logic
- **WebSocket Notifications**: Real-time bidirectional updates
- **In-App Notifications**: Persistent notification storage

### 2. Provider Dashboard Backend (`src/providers/`)

- **Patient Queue Management**: Priority-based queuing with auto-assignment
- **Review Interface**: Approve, reject, escalate, or request additional info
- **Provider Communication**: Secure messaging between providers and patients
- **Emergency Escalation**: Automated alert system with escalation chains

### 3. Consent Management (`src/consent/`)

- **Consent Tracking**: Digital consent with signature support
- **Authorization Service**: Role-based access control
- **Data Sharing Controls**: Granular data sharing policies
- **HIPAA Compliance**: Audit logging and access tracking

### 4. Priority/Severity Routing (`src/routing/`)

- **Emergency Detection**: Keyword-based emergency identification
- **Severity Classification**: Multi-factor severity scoring
- **Provider Matching**: Intelligent provider assignment
- **Escalation Workflows**: Automated escalation based on rules

### 5. Security (`src/security/`)

- **HIPAA Compliance**: Encryption, audit logging, access control
- **Data Encryption**: AES-256 encryption for PHI
- **Access Validation**: Minimum necessary principle enforcement
- **Audit Trail**: Comprehensive logging for compliance

## Installation

```bash
# Install dependencies
npm install

# Copy environment configuration
cp config/.env.example .env

# Edit .env with your credentials
nano .env

# Build the project
npm run build
```

## Configuration

### 1. Email Configuration

```typescript
import { emailConfig } from './config/healthcare-system.config';
import { EmailNotifier } from './src/notifications/email-notifier';

const emailNotifier = new EmailNotifier(emailConfig);
```

### 2. SMS Configuration

```typescript
import { smsConfig } from './config/healthcare-system.config';
import { SMSNotifier } from './src/notifications/sms-notifier';

const smsNotifier = new SMSNotifier(smsConfig);
```

### 3. Complete System Setup

```typescript
import config from './config/healthcare-system.config';
import { NotificationManager } from './src/notifications/notification-manager';
import { PatientQueue } from './src/providers/patient-queue';
import { ProviderService } from './src/providers/provider-service';
import { ConsentManager } from './src/consent/consent-manager';
import { EscalationRouter } from './src/routing/escalation-router';

// Initialize services
const notificationManager = new NotificationManager(config.notifications);
const patientQueue = new PatientQueue(config.queue);
const providerService = new ProviderService();
const consentManager = new ConsentManager();
const escalationRouter = new EscalationRouter();
```

## Usage Examples

### Send Multi-Channel Notification

```typescript
import { NotificationPayload, NotificationChannel, NotificationPriority } from './src/notifications/types';

const notification: NotificationPayload = {
  id: 'notif-123',
  type: 'query_assigned',
  title: 'New Patient Query Assigned',
  message: 'You have been assigned a new urgent patient query',
  priority: NotificationPriority.HIGH,
  channels: [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET, NotificationChannel.INAPP],
  recipient: {
    id: 'provider-456',
    type: 'provider',
    name: 'Dr. Smith',
    email: 'dr.smith@hospital.com',
    preferredChannels: [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET]
  },
  createdAt: new Date(),
  hipaaCompliant: true
};

const results = await notificationManager.sendNotification(notification);
console.log('Notification results:', results);
```

### Manage Patient Queue

```typescript
import { PatientQuery, QueryPriority } from './src/providers/types';

// Add query to queue
const query: PatientQuery = {
  id: 'query-789',
  patientId: 'patient-123',
  patientName: 'John Doe',
  queryType: 'urgent_consultation',
  priority: QueryPriority.URGENT,
  status: QueryStatus.PENDING,
  description: 'Severe chest pain, needs immediate consultation',
  symptoms: ['chest pain', 'shortness of breath'],
  createdAt: new Date(),
  updatedAt: new Date(),
  requiresConsent: true
};

await patientQueue.enqueue(query);

// Get next query
const nextQuery = patientQueue.getNext();

// Assign to provider
await patientQueue.assignToProvider(query.id, 'provider-456');
```

### Check Consent

```typescript
import { ConsentType } from './src/consent/types';

// Check if provider has consent
const hasConsent = consentManager.hasConsentFor(
  'patient-123',
  'provider-456',
  ConsentType.TREATMENT,
  'medical_history'
);

if (hasConsent) {
  // Proceed with treatment
}
```

### Route Query with Severity Detection

```typescript
const routingDecision = await escalationRouter.route(query, availableProviders);

console.log(`Emergency Type: ${routingDecision.emergencyType}`);
console.log(`Severity: ${routingDecision.severity}`);
console.log(`Recommended Provider: ${routingDecision.recommendedProviderId}`);
console.log(`Escalation Required: ${routingDecision.escalationRequired}`);
console.log(`Estimated Response Time: ${routingDecision.estimatedResponseTime} minutes`);
```

### Create Emergency Alert

```typescript
import { EmergencyEscalationService } from './src/providers/emergency-escalation';

const emergencyService = new EmergencyEscalationService();

const alert = await emergencyService.createAlert(
  query,
  'critical',
  'system-auto-detection'
);

// Acknowledge alert
await emergencyService.acknowledgeAlert(alert.id, 'provider-456');

// Resolve alert
await emergencyService.resolveAlert(alert.id, 'provider-456', 'Patient stabilized and admitted');
```

## HIPAA Compliance

### Encryption

All Protected Health Information (PHI) is encrypted using the HIPAA security middleware:

```typescript
import { HIPAASecurityMiddleware } from './src/security/hipaa-security';

const security = new HIPAASecurityMiddleware({
  encryptionKey: process.env.ENCRYPTION_KEY,
  auditEnabled: true
});

// Encrypt sensitive data
const encrypted = security.encrypt(patientData);

// Decrypt when authorized
const decrypted = security.decrypt(encrypted);
```

### Access Validation

```typescript
const accessCheck = security.validateAccess({
  userId: 'provider-456',
  userRole: 'provider',
  resourceId: 'patient-123',
  action: 'read',
  purpose: 'treatment consultation'
});

if (!accessCheck.authorized) {
  console.error('Access denied:', accessCheck.reason);
}
```

### Audit Logging

All data access is automatically logged for HIPAA compliance:

```typescript
import { HIPAAComplianceService } from './src/consent/hipaa-compliance';

const hipaaService = new HIPAAComplianceService();

// Log access
await hipaaService.logAccess({
  timestamp: new Date(),
  userId: 'provider-456',
  userType: 'provider',
  action: 'read',
  resource: 'medical_record',
  patientId: 'patient-123',
  authorized: true
});

// Get access logs
const logs = hipaaService.getAccessLogs('patient-123', {
  startDate: new Date('2025-01-01'),
  endDate: new Date()
});
```

## Testing

```bash
# Run all tests
npm test

# Run specific test suite
npm test -- notifications
npm test -- providers
npm test -- consent
npm test -- routing
```

## Security Best Practices

1. **Never commit secrets**: Use environment variables for all credentials
2. **Enable encryption**: Always encrypt PHI in transit and at rest
3. **Audit logging**: Enable comprehensive audit logging for compliance
4. **Access control**: Implement role-based access control (RBAC)
5. **Session management**: Use short session timeouts (30 minutes recommended)
6. **MFA**: Enable multi-factor authentication for provider access
7. **Regular audits**: Perform regular HIPAA compliance audits

## Monitoring

### System Statistics

```typescript
// Notification statistics
const notifStats = notificationManager.getStats();
console.log(`Total sent: ${notifStats.totalSent}`);
console.log(`Success rate: ${(notifStats.successRate * 100).toFixed(2)}%`);

// Queue statistics
const queueStats = patientQueue.getStats();
console.log(`Queue size: ${queueStats.totalQueries}`);
console.log(`Average wait time: ${queueStats.averageWaitTime.toFixed(1)} minutes`);

// Provider statistics
const providerStats = providerService.getStats();
console.log(`Online providers: ${providerStats.onlineProviders}`);
console.log(`Average case load: ${providerStats.averageCaseLoad.toFixed(1)}`);

// HIPAA compliance statistics
const hipaaStats = hipaaService.getStats();
console.log(`Compliant patients: ${hipaaStats.compliantPatients}`);
console.log(`Unauthorized attempts: ${hipaaStats.unauthorizedAttempts}`);
```

## Architecture

```
src/
├── notifications/          # Multi-channel notification system
│   ├── types.ts           # Type definitions
│   ├── email-notifier.ts  # Email delivery
│   ├── sms-notifier.ts    # SMS delivery
│   ├── webhook-notifier.ts # Webhook delivery
│   ├── websocket-notifier.ts # WebSocket real-time
│   ├── inapp-notifier.ts  # In-app notifications
│   └── notification-manager.ts # Orchestration
├── providers/             # Provider dashboard backend
│   ├── types.ts          # Type definitions
│   ├── provider-service.ts # Provider management
│   ├── patient-queue.ts  # Queue management
│   ├── review-interface.ts # Review workflows
│   ├── provider-communication.ts # Messaging
│   └── emergency-escalation.ts # Emergency alerts
├── consent/              # Consent management
│   ├── types.ts         # Type definitions
│   ├── consent-manager.ts # Consent lifecycle
│   ├── authorization-service.ts # Access control
│   ├── data-sharing-controls.ts # Sharing policies
│   └── hipaa-compliance.ts # HIPAA tracking
├── routing/             # Priority/severity routing
│   ├── types.ts        # Type definitions
│   ├── emergency-detector.ts # Emergency detection
│   ├── severity-classifier.ts # Severity scoring
│   ├── provider-matcher.ts # Provider matching
│   └── escalation-router.ts # Routing decisions
└── security/           # Security middleware
    └── hipaa-security.ts # HIPAA compliance

config/                 # Configuration files
docs/                  # Documentation
tests/                 # Test suites
```

## Support

For issues, questions, or contributions, please refer to the main project repository.

## License

This healthcare system implementation follows HIPAA security and privacy requirements.
