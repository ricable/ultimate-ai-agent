/**
 * Ericsson ENM CM Writer
 *
 * Configuration Management writer for ENM operations
 * Supports bulk operations and transaction management
 *
 * @module cm-writer
 */

import {
  EUtranCellFDD,
  NRCellDU,
  AntennaUnit,
  momGenerator,
  EUtranCellFDDSchema,
  NRCellDUSchema,
  AntennaUnitSchema
} from './mom-schema.js';

// ============================================================================
// Types
// ============================================================================

export type ManagedObjectType = 'EUtranCellFDD' | 'NRCellDU' | 'AntennaUnit';
export type CMOperation = 'create' | 'update' | 'delete';

export interface CMChangeRequest {
  id: string;
  type: ManagedObjectType;
  operation: CMOperation;
  data: Partial<EUtranCellFDD | NRCellDU | AntennaUnit>;
  priority?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  scheduledTime?: Date;
  rollbackOnError?: boolean;
}

export interface CMTransaction {
  transactionId: string;
  changes: CMChangeRequest[];
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED' | 'ROLLED_BACK';
  startTime?: Date;
  endTime?: Date;
  error?: string;
  rollbackData?: any[];
}

export interface CMWriteResult {
  success: boolean;
  transactionId: string;
  appliedChanges: number;
  failedChanges: number;
  warnings: string[];
  errors: string[];
  executionTime: number; // milliseconds
  xml?: string; // Generated XML for review
}

export interface ENMConnectionConfig {
  host: string;
  port: number;
  username: string;
  password: string;
  useHTTPS: boolean;
  timeout: number; // milliseconds
  maxRetries: number;
}

// ============================================================================
// CM Writer Class
// ============================================================================

/**
 * ENM Configuration Management Writer
 * Handles bulk operations and transaction management
 */
export class ENMCMWriter {
  private config: ENMConnectionConfig;
  private transactions: Map<string, CMTransaction> = new Map();
  private transactionCounter: number = 0;

  constructor(config: ENMConnectionConfig) {
    this.config = config;
  }

  /**
   * Create a new CM transaction
   */
  createTransaction(changes: CMChangeRequest[]): string {
    const transactionId = `TX-${Date.now()}-${++this.transactionCounter}`;

    const transaction: CMTransaction = {
      transactionId,
      changes: [...changes], // Clone to prevent external modification
      status: 'PENDING',
      rollbackData: [],
    };

    this.transactions.set(transactionId, transaction);
    return transactionId;
  }

  /**
   * Execute a CM transaction
   */
  async executeTransaction(transactionId: string): Promise<CMWriteResult> {
    const startTime = Date.now();
    const transaction = this.transactions.get(transactionId);

    if (!transaction) {
      throw new Error(`Transaction ${transactionId} not found`);
    }

    const result: CMWriteResult = {
      success: false,
      transactionId,
      appliedChanges: 0,
      failedChanges: 0,
      warnings: [],
      errors: [],
      executionTime: 0,
    };

    try {
      transaction.status = 'IN_PROGRESS';
      transaction.startTime = new Date();

      // Validate all changes before applying
      const validationErrors = this.validateChanges(transaction.changes);
      if (validationErrors.length > 0) {
        result.errors.push(...validationErrors);
        transaction.status = 'FAILED';
        transaction.error = validationErrors.join('; ');
        result.executionTime = Date.now() - startTime;
        return result;
      }

      // Generate XML for all changes
      const xml = this.generateBulkXML(transaction.changes);
      result.xml = xml;

      // Apply changes (in parallel for performance)
      const changeResults = await Promise.allSettled(
        transaction.changes.map((change) => this.applyChange(change))
      );

      // Process results
      changeResults.forEach((changeResult, index) => {
        if (changeResult.status === 'fulfilled') {
          result.appliedChanges++;
          transaction.rollbackData?.push(changeResult.value.rollbackData);
        } else {
          result.failedChanges++;
          result.errors.push(
            `Change ${transaction.changes[index].id}: ${changeResult.reason}`
          );
        }
      });

      // Check if transaction should be rolled back
      if (result.failedChanges > 0) {
        const shouldRollback = transaction.changes.some(c => c.rollbackOnError !== false);

        if (shouldRollback) {
          await this.rollbackTransaction(transactionId);
          transaction.status = 'ROLLED_BACK';
          result.warnings.push('Transaction rolled back due to errors');
        } else {
          transaction.status = 'COMPLETED';
          result.warnings.push('Transaction completed with partial failures');
        }
      } else {
        transaction.status = 'COMPLETED';
        result.success = true;
      }

      transaction.endTime = new Date();
      result.executionTime = Date.now() - startTime;

      return result;

    } catch (error) {
      transaction.status = 'FAILED';
      transaction.error = error instanceof Error ? error.message : String(error);
      result.errors.push(transaction.error);
      result.executionTime = Date.now() - startTime;

      return result;
    }
  }

  /**
   * Validate changes before applying
   */
  private validateChanges(changes: CMChangeRequest[]): string[] {
    const errors: string[] = [];

    changes.forEach((change, index) => {
      try {
        switch (change.type) {
          case 'EUtranCellFDD':
            EUtranCellFDDSchema.partial().parse(change.data);
            break;
          case 'NRCellDU':
            NRCellDUSchema.partial().parse(change.data);
            break;
          case 'AntennaUnit':
            AntennaUnitSchema.partial().parse(change.data);
            break;
          default:
            errors.push(`Change ${index}: Unknown type ${change.type}`);
        }

        // Validate 3GPP constraints
        this.validate3GPPConstraints(change, errors);

      } catch (error) {
        if (error instanceof Error) {
          errors.push(`Change ${index} (${change.id}): ${error.message}`);
        }
      }
    });

    return errors;
  }

  /**
   * Validate 3GPP TS 28.552 constraints
   */
  private validate3GPPConstraints(change: CMChangeRequest, errors: string[]): void {
    const data = change.data as any;

    // Power constraints (3GPP TS 28.552)
    if (data.maxTxPower !== undefined) {
      if (data.maxTxPower < -130 || data.maxTxPower > 46) {
        errors.push(`${change.id}: maxTxPower must be between -130 and 46 dBm`);
      }
    }

    // Tilt constraints
    if (data.mechanicalTilt !== undefined && data.electricalTilt !== undefined) {
      const totalTilt = data.mechanicalTilt + data.electricalTilt;
      if (totalTilt > 30) {
        errors.push(`${change.id}: Total tilt (${totalTilt}°) exceeds maximum of 30°`);
      }
    }

    // LTE-specific constraints
    if (change.type === 'EUtranCellFDD') {
      if (data.p0NominalPUSCH !== undefined) {
        if (data.p0NominalPUSCH < -130 || data.p0NominalPUSCH > -70) {
          errors.push(`${change.id}: p0NominalPUSCH must be between -130 and -70 dBm`);
        }
      }

      if (data.alpha !== undefined) {
        const validAlpha = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        if (!validAlpha.includes(data.alpha)) {
          errors.push(`${change.id}: alpha must be one of ${validAlpha.join(', ')}`);
        }
      }
    }

    // 5G NR-specific constraints
    if (change.type === 'NRCellDU') {
      if (data.pZeroNomPusch !== undefined) {
        if (data.pZeroNomPusch < -202 || data.pZeroNomPusch > 24) {
          errors.push(`${change.id}: pZeroNomPusch must be between -202 and 24 dBm`);
        }
      }
    }
  }

  /**
   * Generate bulk XML for changes
   */
  private generateBulkXML(changes: CMChangeRequest[]): string {
    const objects = changes.map((change) => ({
      type: change.type,
      data: change.data,
      operation: change.operation,
    }));

    return momGenerator.generateBulkXML(objects);
  }

  /**
   * Apply a single change (simulated - would call ENM API in production)
   */
  private async applyChange(change: CMChangeRequest): Promise<{
    success: boolean;
    rollbackData: any;
  }> {
    // Simulate network delay
    await this.delay(10 + Math.random() * 50);

    // In production, this would:
    // 1. Connect to ENM via HTTPS/REST API
    // 2. Authenticate using credentials
    // 3. Submit XML via Bulk CM Import
    // 4. Poll for job completion
    // 5. Return results

    // For now, simulate success with 95% probability
    const success = Math.random() > 0.05;

    if (!success) {
      throw new Error(`Failed to apply change ${change.id}: Network timeout`);
    }

    // Store rollback data (previous values)
    const rollbackData = {
      id: change.id,
      type: change.type,
      operation: this.getRollbackOperation(change.operation),
      data: { ...change.data }, // Clone current state
    };

    return { success, rollbackData };
  }

  /**
   * Rollback a transaction
   */
  private async rollbackTransaction(transactionId: string): Promise<void> {
    const transaction = this.transactions.get(transactionId);

    if (!transaction || !transaction.rollbackData) {
      throw new Error(`Cannot rollback transaction ${transactionId}`);
    }

    // Apply rollback changes in reverse order
    const rollbackChanges = transaction.rollbackData.reverse();

    await Promise.all(
      rollbackChanges.map((rollback) => this.applyChange(rollback as CMChangeRequest))
    );
  }

  /**
   * Get rollback operation for a given operation
   */
  private getRollbackOperation(operation: CMOperation): CMOperation {
    switch (operation) {
      case 'create': return 'delete';
      case 'delete': return 'create';
      case 'update': return 'update'; // Restore previous values
    }
  }

  /**
   * Write single object (convenience method)
   */
  async writeSingle(
    type: ManagedObjectType,
    operation: CMOperation,
    data: Partial<EUtranCellFDD | NRCellDU | AntennaUnit>
  ): Promise<CMWriteResult> {
    const change: CMChangeRequest = {
      id: `${type}-${Date.now()}`,
      type,
      operation,
      data,
      rollbackOnError: true,
    };

    const txId = this.createTransaction([change]);
    return this.executeTransaction(txId);
  }

  /**
   * Write bulk changes
   */
  async writeBulk(changes: CMChangeRequest[]): Promise<CMWriteResult> {
    const txId = this.createTransaction(changes);
    return this.executeTransaction(txId);
  }

  /**
   * Get transaction status
   */
  getTransactionStatus(transactionId: string): CMTransaction | undefined {
    return this.transactions.get(transactionId);
  }

  /**
   * Cancel pending transaction
   */
  cancelTransaction(transactionId: string): boolean {
    const transaction = this.transactions.get(transactionId);

    if (!transaction) {
      return false;
    }

    if (transaction.status === 'PENDING') {
      this.transactions.delete(transactionId);
      return true;
    }

    return false; // Cannot cancel in-progress or completed transactions
  }

  /**
   * Utility: delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Export XML for manual review/import
   */
  exportXML(changes: CMChangeRequest[]): string {
    return this.generateBulkXML(changes);
  }

  /**
   * Validate connectivity to ENM
   */
  async testConnection(): Promise<boolean> {
    try {
      // In production, this would:
      // 1. Connect to ENM API endpoint
      // 2. Authenticate
      // 3. Execute test query

      await this.delay(100);
      return true;

    } catch (error) {
      console.error('ENM connection test failed:', error);
      return false;
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create ENM CM Writer instance
 */
export function createCMWriter(config: Partial<ENMConnectionConfig> = {}): ENMCMWriter {
  const defaultConfig: ENMConnectionConfig = {
    host: process.env.ENM_HOST || 'enm.ericsson.local',
    port: parseInt(process.env.ENM_PORT || '443'),
    username: process.env.ENM_USERNAME || 'admin',
    password: process.env.ENM_PASSWORD || '',
    useHTTPS: true,
    timeout: 30000, // 30 seconds
    maxRetries: 3,
    ...config,
  };

  return new ENMCMWriter(defaultConfig);
}

/**
 * Create batch change request helper
 */
export function createBatchChanges(
  type: ManagedObjectType,
  operation: CMOperation,
  dataList: Array<Partial<EUtranCellFDD | NRCellDU | AntennaUnit>>
): CMChangeRequest[] {
  return dataList.map((data, index) => ({
    id: `${type}-${Date.now()}-${index}`,
    type,
    operation,
    data,
    priority: 'MEDIUM',
    rollbackOnError: true,
  }));
}

// ============================================================================
// Export default instance
// ============================================================================

export const cmWriter = createCMWriter();
