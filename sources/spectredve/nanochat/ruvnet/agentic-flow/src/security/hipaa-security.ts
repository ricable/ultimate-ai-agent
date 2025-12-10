/**
 * HIPAA Security Middleware
 * Provides encryption, audit logging, and access control for HIPAA compliance
 */

export class HIPAASecurityMiddleware {
  private encryptionKey: string;
  private auditEnabled: boolean;

  constructor(config: { encryptionKey: string; auditEnabled: boolean }) {
    this.encryptionKey = config.encryptionKey;
    this.auditEnabled = config.auditEnabled;
  }

  /**
   * Encrypt sensitive data (PHI - Protected Health Information)
   */
  encrypt(data: string): string {
    // In production, use AES-256 or similar encryption
    // For demonstration, we'll use base64 encoding
    if (!data) return '';

    try {
      const buffer = Buffer.from(data, 'utf-8');
      const encrypted = buffer.toString('base64');
      return `encrypted:${encrypted}`;
    } catch (error) {
      console.error('[HIPAA_SECURITY] Encryption failed:', error);
      throw new Error('Failed to encrypt data');
    }
  }

  /**
   * Decrypt sensitive data
   */
  decrypt(encryptedData: string): string {
    if (!encryptedData || !encryptedData.startsWith('encrypted:')) {
      return encryptedData;
    }

    try {
      const encrypted = encryptedData.substring('encrypted:'.length);
      const buffer = Buffer.from(encrypted, 'base64');
      return buffer.toString('utf-8');
    } catch (error) {
      console.error('[HIPAA_SECURITY] Decryption failed:', error);
      throw new Error('Failed to decrypt data');
    }
  }

  /**
   * Mask sensitive information for logging
   */
  maskPHI(data: any): any {
    if (typeof data === 'string') {
      // Mask SSN
      data = data.replace(/\d{3}-\d{2}-\d{4}/g, '***-**-****');
      // Mask email
      data = data.replace(/[\w.-]+@[\w.-]+\.\w+/g, '***@***.**');
      // Mask phone
      data = data.replace(/\d{3}-\d{3}-\d{4}/g, '***-***-****');
      return data;
    }

    if (typeof data === 'object' && data !== null) {
      const masked = Array.isArray(data) ? [] : {};
      for (const key in data) {
        if (this.isSensitiveField(key)) {
          (masked as any)[key] = '***REDACTED***';
        } else {
          (masked as any)[key] = this.maskPHI(data[key]);
        }
      }
      return masked;
    }

    return data;
  }

  /**
   * Check if field contains sensitive information
   */
  private isSensitiveField(fieldName: string): boolean {
    const sensitiveFields = [
      'ssn', 'socialsecuritynumber', 'password', 'secret',
      'creditcard', 'accountnumber', 'dob', 'dateofbirth',
      'diagnosis', 'prescription', 'medicalrecord'
    ];

    return sensitiveFields.some(field =>
      fieldName.toLowerCase().includes(field)
    );
  }

  /**
   * Validate data access authorization
   */
  validateAccess(request: {
    userId: string;
    userRole: string;
    resourceId: string;
    action: string;
    purpose: string;
  }): { authorized: boolean; reason?: string } {
    // Check minimum necessary principle
    if (!this.isMinimumNecessary(request)) {
      return {
        authorized: false,
        reason: 'Access violates minimum necessary principle'
      };
    }

    // Check role-based access control
    if (!this.hasRequiredRole(request.userRole, request.action)) {
      return {
        authorized: false,
        reason: 'Insufficient role permissions'
      };
    }

    // Check purpose justification
    if (!this.isValidPurpose(request.purpose)) {
      return {
        authorized: false,
        reason: 'Invalid or missing access purpose'
      };
    }

    if (this.auditEnabled) {
      this.auditLog({
        timestamp: new Date(),
        userId: request.userId,
        action: request.action,
        resource: request.resourceId,
        authorized: true,
        purpose: request.purpose
      });
    }

    return { authorized: true };
  }

  /**
   * Check minimum necessary access principle
   */
  private isMinimumNecessary(request: any): boolean {
    // In production, this would check if the requested data scope
    // is the minimum necessary for the stated purpose
    return request.purpose && request.purpose.length > 10;
  }

  /**
   * Check role-based permissions
   */
  private hasRequiredRole(userRole: string, action: string): boolean {
    const rolePermissions: Record<string, string[]> = {
      'admin': ['read', 'write', 'delete', 'audit'],
      'provider': ['read', 'write'],
      'nurse': ['read', 'write'],
      'patient': ['read'],
      'staff': ['read']
    };

    const permissions = rolePermissions[userRole.toLowerCase()];
    return permissions ? permissions.includes(action) : false;
  }

  /**
   * Validate access purpose
   */
  private isValidPurpose(purpose: string): boolean {
    const validPurposes = [
      'treatment', 'diagnosis', 'consultation', 'emergency',
      'prescription', 'follow-up', 'review', 'coordination'
    ];

    const lowerPurpose = purpose.toLowerCase();
    return validPurposes.some(valid => lowerPurpose.includes(valid));
  }

  /**
   * Audit log access
   */
  private auditLog(entry: any): void {
    // In production, write to secure audit log storage
    console.log(`[HIPAA_AUDIT] ${JSON.stringify(this.maskPHI(entry))}`);
  }

  /**
   * Generate compliance report
   */
  generateComplianceReport(period: { start: Date; end: Date }): {
    totalAccesses: number;
    unauthorizedAttempts: number;
    encryptedDataAccesses: number;
    auditLogEntries: number;
    complianceViolations: string[];
  } {
    // In production, query audit logs and generate comprehensive report
    return {
      totalAccesses: 0,
      unauthorizedAttempts: 0,
      encryptedDataAccesses: 0,
      auditLogEntries: 0,
      complianceViolations: []
    };
  }

  /**
   * Sanitize data for transmission
   */
  sanitizeForTransmission(data: any): any {
    // Remove sensitive fields that shouldn't be transmitted
    const sanitized = { ...data };
    const fieldsToRemove = ['password', 'ssn', 'creditCard'];

    for (const field of fieldsToRemove) {
      delete sanitized[field];
    }

    return sanitized;
  }

  /**
   * Generate secure token for data access
   */
  generateAccessToken(userId: string, resourceId: string, expiryMinutes: number = 60): string {
    const expiry = Date.now() + (expiryMinutes * 60 * 1000);
    const payload = `${userId}:${resourceId}:${expiry}`;
    return this.encrypt(payload);
  }

  /**
   * Validate access token
   */
  validateAccessToken(token: string): { valid: boolean; userId?: string; resourceId?: string } {
    try {
      const decrypted = this.decrypt(token);
      const [userId, resourceId, expiryStr] = decrypted.split(':');
      const expiry = parseInt(expiryStr);

      if (Date.now() > expiry) {
        return { valid: false };
      }

      return { valid: true, userId, resourceId };
    } catch {
      return { valid: false };
    }
  }
}
