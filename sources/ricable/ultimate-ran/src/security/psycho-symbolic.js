/**
 * Psycho-Symbolic Integration
 * The "Firewall" between Neural Agents and the Network
 *
 * While Neural agents operate on Intuition (Probabilistic),
 * the Governance layer operates on Rules (Deterministic).
 */

import { readFileSync } from 'fs';

export class PsychoSymbolicGuard {
  constructor({ constraintsPath }) {
    this.constraintsPath = constraintsPath || './config/constraints/ran-physics.json';
    this.constraints = this.loadConstraints();
    this.auditLog = [];
  }

  loadConstraints() {
    try {
      const data = readFileSync(this.constraintsPath, 'utf-8');
      return JSON.parse(data);
    } catch (error) {
      console.warn('[PSI-GUARD] Failed to load constraints, using defaults');
      return this.getDefaultConstraints();
    }
  }

  getDefaultConstraints() {
    return {
      '3gpp_constraints': {
        bler_max: 0.1,
        power_max_dbm: 46,
        rsrp_min_dbm: -140
      },
      'hardware_limits': {
        radio_6630: { tx_power_max_dbm: 46 }
      }
    };
  }

  /**
   * Intercept and validate a command before execution
   * Returns ALLOW, BLOCK, or REQUIRE_APPROVAL
   */
  async interceptCommand(command) {
    console.log(`[PSI-GUARD] Intercepting command: ${command.tool}.${command.action}`);

    const audit = {
      timestamp: new Date().toISOString(),
      command,
      result: null,
      violations: []
    };

    // Check against 3GPP constraints
    const gppViolations = this.check3GPPConstraints(command);
    audit.violations.push(...gppViolations);

    // Check against hardware limits
    const hwViolations = this.checkHardwareLimits(command);
    audit.violations.push(...hwViolations);

    // Check if this is a critical action requiring approval
    if (this.isCriticalAction(command)) {
      audit.result = 'REQUIRE_APPROVAL';
      this.auditLog.push(audit);
      return { status: 'REQUIRE_APPROVAL', command, reason: 'Critical action' };
    }

    // Determine final result
    if (audit.violations.length > 0) {
      const hasCritical = audit.violations.some(v => v.severity === 'CRITICAL');
      audit.result = hasCritical ? 'BLOCK' : 'WARN';
    } else {
      audit.result = 'ALLOW';
    }

    this.auditLog.push(audit);

    if (audit.result === 'BLOCK') {
      console.error(`[PSI-GUARD] BLOCKED: ${audit.violations.map(v => v.message).join(', ')}`);
      return { status: 'VIOLATION', command, violations: audit.violations };
    }

    if (audit.result === 'WARN') {
      console.warn(`[PSI-GUARD] WARNING: ${audit.violations.map(v => v.message).join(', ')}`);
    }

    return { status: 'ALLOW', command };
  }

  /**
   * Check command against 3GPP constraints
   */
  check3GPPConstraints(command) {
    const violations = [];
    const constraints = this.constraints['3gpp_constraints'] || {};

    // Check power limits
    if (command.params?.power !== undefined) {
      const maxPower = constraints.power_max_dbm || 46;
      if (command.params.power > maxPower) {
        violations.push({
          type: 'POWER_LIMIT',
          severity: 'CRITICAL',
          message: `Power ${command.params.power} dBm exceeds max ${maxPower} dBm`,
          constraint: '3GPP TS 38.104'
        });
      }
    }

    // Check BLER limits
    if (command.params?.targetBler !== undefined) {
      const maxBler = constraints.bler_max || 0.1;
      if (command.params.targetBler > maxBler) {
        violations.push({
          type: 'BLER_LIMIT',
          severity: 'HIGH',
          message: `Target BLER ${command.params.targetBler} exceeds max ${maxBler}`,
          constraint: '3GPP TS 38.331'
        });
      }
    }

    // Check cellIndividualOffset limits
    if (command.params?.cio !== undefined) {
      const maxCIO = 24; // dB
      if (Math.abs(command.params.cio) > maxCIO) {
        violations.push({
          type: 'CIO_LIMIT',
          severity: 'MEDIUM',
          message: `CIO ${command.params.cio} dB exceeds range +/- ${maxCIO} dB`,
          constraint: '3GPP TS 38.331'
        });
      }
    }

    return violations;
  }

  /**
   * Check command against hardware limits
   */
  checkHardwareLimits(command) {
    const violations = [];
    const hwLimits = this.constraints.hardware_limits || {};

    // Get hardware type from command context
    const hwType = command.hardware || 'radio_6630';
    const limits = hwLimits[hwType];

    if (!limits) {
      return violations;
    }

    // Check TX power
    if (command.params?.txPower !== undefined) {
      const maxTxPower = limits.tx_power_max_dbm || 46;
      if (command.params.txPower > maxTxPower) {
        violations.push({
          type: 'HW_TX_POWER',
          severity: 'CRITICAL',
          message: `TX Power ${command.params.txPower} dBm exceeds ${hwType} max ${maxTxPower} dBm`,
          hardware: hwType
        });
      }
    }

    return violations;
  }

  /**
   * Determine if action requires human approval
   */
  isCriticalAction(command) {
    const criticalActions = [
      'cell_lock',
      'frequency_retune',
      'reboot_bbu',
      'power_override',
      'configuration_rollback',
      'antenna_tilt_major', // > 5 degrees
      'sector_shutdown'
    ];

    return criticalActions.includes(command.action);
  }

  /**
   * Verify a signed approval token
   */
  verifyApprovalToken(token, command) {
    console.log(`[PSI-GUARD] Verifying approval token for: ${command.action}`);

    // In production, verifies cryptographic signature (YubiKey, biometric)
    if (!token || !token.signature) {
      return { valid: false, reason: 'Missing signature' };
    }

    // Check token hasn't expired
    if (token.expiresAt && new Date(token.expiresAt) < new Date()) {
      return { valid: false, reason: 'Token expired' };
    }

    // Verify signature matches command
    // In production, uses ML-DSA (Module-Lattice-Based Digital Signature)
    const signatureValid = this.verifyMLDSA(token.signature, command);

    return {
      valid: signatureValid,
      reason: signatureValid ? 'Signature verified' : 'Invalid signature'
    };
  }

  verifyMLDSA(signature, command) {
    // Placeholder for ML-DSA verification
    // In production, uses quantum-resistant signature verification
    return signature && signature.length > 0;
  }

  /**
   * Get audit log
   */
  getAuditLog(limit = 100) {
    return this.auditLog.slice(-limit);
  }

  /**
   * Generate compliance report
   */
  generateComplianceReport() {
    const total = this.auditLog.length;
    const blocked = this.auditLog.filter(a => a.result === 'BLOCK').length;
    const warnings = this.auditLog.filter(a => a.result === 'WARN').length;
    const approvals = this.auditLog.filter(a => a.result === 'REQUIRE_APPROVAL').length;

    return {
      total,
      allowed: total - blocked - warnings - approvals,
      blocked,
      warnings,
      pendingApprovals: approvals,
      complianceRate: ((total - blocked) / total * 100).toFixed(2) + '%',
      generatedAt: new Date().toISOString()
    };
  }
}

export { PsychoSymbolicGuard };
