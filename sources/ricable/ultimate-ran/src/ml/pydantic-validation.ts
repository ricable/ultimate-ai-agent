/**
 * PydanticAI-Style Validation for ML Outputs
 *
 * Provides runtime validation and type safety for ML model outputs using
 * a Pydantic-inspired validation framework. Ensures that RAN optimizations
 * from ML models comply with 3GPP constraints and physical limits before
 * being applied to the live network.
 *
 * Key Features:
 * - Type-safe schema validation
 * - 3GPP TS 38.331 and TS 28.552 compliance checks
 * - Physics-based constraint validation
 * - Automatic coercion and sanitization
 * - Detailed error reporting
 *
 * @module ml/pydantic-validation
 * @version 7.0.0-alpha.1
 */

import type { CMParameters, PMCounters } from '../learning/self-learner.js';
import type { Recommendation } from './ruvector-gnn.js';
import type { PropagationResult } from './attention-gnn.js';

// ============================================================================
// Validation Schema Definitions
// ============================================================================

/**
 * Validation error
 */
export interface ValidationError {
  field: string;
  message: string;
  constraint: string;
  value: any;
  expected?: any;
}

/**
 * Validation result
 */
export interface ValidationResult<T> {
  valid: boolean;
  data?: T;
  errors: ValidationError[];
  warnings?: string[];
}

/**
 * Field validator function
 */
export type FieldValidator<T> = (value: T, context?: any) => ValidationError | null;

/**
 * Schema field definition
 */
export interface SchemaField<T = any> {
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  required?: boolean;
  validators?: FieldValidator<T>[];
  min?: number;
  max?: number;
  default?: T;
  description?: string;
  units?: string;
  coerce?: (value: any) => T;
}

/**
 * Validation schema
 */
export interface ValidationSchema<T = any> {
  fields: Record<keyof T, SchemaField>;
  customValidators?: Array<(data: T) => ValidationError | null>;
}

// ============================================================================
// 3GPP Constraint Validators
// ============================================================================

/**
 * 3GPP TS 38.331 validators for CM parameters
 */
export class ThreeGPPValidators {
  /**
   * P0-NominalPUSCH validation (3GPP TS 38.331 section 6.3.2)
   * Range: -202 to 24 dBm (in 1 dB steps)
   */
  static validateP0NominalPUSCH: FieldValidator<number> = (value) => {
    if (value < -202 || value > 24) {
      return {
        field: 'p0NominalPUSCH',
        message: 'P0-NominalPUSCH must be between -202 and 24 dBm',
        constraint: '3GPP TS 38.331 section 6.3.2',
        value,
        expected: '[-202, 24]'
      };
    }

    // Must be integer (1 dB steps)
    if (!Number.isInteger(value)) {
      return {
        field: 'p0NominalPUSCH',
        message: 'P0-NominalPUSCH must be in 1 dB steps (integer)',
        constraint: '3GPP TS 38.331',
        value,
        expected: 'integer'
      };
    }

    return null;
  };

  /**
   * Alpha (path loss compensation factor) validation
   * Range: {0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
   */
  static validateAlpha: FieldValidator<number> = (value) => {
    const validValues = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    if (!validValues.includes(value)) {
      return {
        field: 'alpha',
        message: 'Alpha must be one of the 3GPP-defined values',
        constraint: '3GPP TS 38.331',
        value,
        expected: validValues
      };
    }

    return null;
  };

  /**
   * Electrical tilt validation
   * Range: 0 to 15 degrees (typical for remote electrical tilt)
   */
  static validateElectricalTilt: FieldValidator<number> = (value) => {
    if (value < 0 || value > 15) {
      return {
        field: 'electricalTilt',
        message: 'Electrical tilt must be between 0 and 15 degrees',
        constraint: 'RET antenna specifications',
        value,
        expected: '[0, 15]'
      };
    }

    // Must be in 0.1 degree steps
    if (Math.round(value * 10) !== value * 10) {
      return {
        field: 'electricalTilt',
        message: 'Electrical tilt must be in 0.1 degree steps',
        constraint: 'RET precision',
        value,
        expected: '0.1 degree precision'
      };
    }

    return null;
  };

  /**
   * TX power validation
   * Range: -130 to 46 dBm (3GPP spec)
   */
  static validateTxPower: FieldValidator<number> = (value) => {
    if (value < -130 || value > 46) {
      return {
        field: 'txPower',
        message: 'TX power must be between -130 and 46 dBm',
        constraint: '3GPP TS 38.104',
        value,
        expected: '[-130, 46]'
      };
    }

    return null;
  };

  /**
   * Beam weights validation (beamforming)
   * Must sum to 1.0 and all be non-negative
   */
  static validateBeamWeights: FieldValidator<number[]> = (value) => {
    if (!Array.isArray(value) || value.length === 0) {
      return {
        field: 'beamWeights',
        message: 'Beam weights must be a non-empty array',
        constraint: 'Beamforming specification',
        value,
        expected: 'non-empty array'
      };
    }

    // Check non-negative
    for (let i = 0; i < value.length; i++) {
      if (value[i] < 0) {
        return {
          field: `beamWeights[${i}]`,
          message: 'Beam weight cannot be negative',
          constraint: 'Physical constraint',
          value: value[i],
          expected: '>= 0'
        };
      }
    }

    // Check sum to 1.0 (within tolerance)
    const sum = value.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1.0) > 0.01) {
      return {
        field: 'beamWeights',
        message: 'Beam weights must sum to 1.0',
        constraint: 'Power normalization',
        value: sum,
        expected: 1.0
      };
    }

    return null;
  };
}

// ============================================================================
// Physics-Based Validators
// ============================================================================

/**
 * Physics-based constraint validators
 */
export class PhysicsValidators {
  /**
   * SINR feasibility check
   * Ensure predicted SINR is physically possible given interference
   */
  static validateSINRFeasibility: FieldValidator<number> = (value, context) => {
    // SINR range: typically -20 to 30 dB
    if (value < -30 || value > 40) {
      return {
        field: 'sinr',
        message: 'SINR value is outside physically feasible range',
        constraint: 'Shannon capacity limit',
        value,
        expected: '[-30, 40] dB'
      };
    }

    return null;
  };

  /**
   * Power budget validation
   * Ensure total power doesn't exceed PA capability
   */
  static validatePowerBudget: FieldValidator<number> = (value, context) => {
    const maxPAPower = context?.maxPAPower || 46;  // dBm

    if (value > maxPAPower) {
      return {
        field: 'totalPower',
        message: 'Total power exceeds PA capability',
        constraint: 'Power amplifier limit',
        value,
        expected: `<= ${maxPAPower} dBm`
      };
    }

    return null;
  };

  /**
   * Thermal noise floor validation
   */
  static validateNoiseFloor: FieldValidator<number> = (value) => {
    // Thermal noise floor: -174 dBm/Hz + NF
    const minNoiseFloor = -174 + 9;  // -165 dBm/Hz with 9 dB NF

    if (value < minNoiseFloor) {
      return {
        field: 'noiseFloor',
        message: 'Noise floor below thermal limit',
        constraint: 'Thermodynamics',
        value,
        expected: `>= ${minNoiseFloor} dBm/Hz`
      };
    }

    return null;
  };

  /**
   * Interference coupling validation
   * Ensure interference pattern is consistent with geometry
   */
  static validateInterferenceCoupling: FieldValidator<number> = (value, context) => {
    const distance = context?.distance;  // meters
    const frequency = context?.frequency || 2100;  // MHz

    if (distance) {
      // Free space path loss
      const fspl = 20 * Math.log10(distance) + 20 * Math.log10(frequency) + 32.44;

      // Coupling loss should be >= FSPL
      if (value < fspl - 20) {  // Allow 20 dB margin for gains
        return {
          field: 'couplingLoss',
          message: 'Coupling loss violates free space path loss',
          constraint: 'Friis transmission equation',
          value,
          expected: `>= ${fspl - 20} dB`
        };
      }
    }

    return null;
  };
}

// ============================================================================
// Schema Definitions for ML Outputs
// ============================================================================

/**
 * CM Parameters validation schema
 */
export const CMParametersSchema: ValidationSchema<CMParameters> = {
  fields: {
    p0NominalPUSCH: {
      type: 'number',
      required: false,
      validators: [ThreeGPPValidators.validateP0NominalPUSCH],
      min: -202,
      max: 24,
      description: 'P0 nominal for PUSCH power control',
      units: 'dBm',
      coerce: (val) => Math.round(val)  // Coerce to integer
    },
    alpha: {
      type: 'number',
      required: false,
      validators: [ThreeGPPValidators.validateAlpha],
      description: 'Path loss compensation factor',
      units: 'dimensionless',
      coerce: (val) => {
        // Coerce to nearest valid value
        const validValues = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        return validValues.reduce((prev, curr) =>
          Math.abs(curr - val) < Math.abs(prev - val) ? curr : prev
        );
      }
    },
    electricalTilt: {
      type: 'number',
      required: false,
      validators: [ThreeGPPValidators.validateElectricalTilt],
      min: 0,
      max: 15,
      description: 'Remote electrical tilt angle',
      units: 'degrees',
      coerce: (val) => Math.round(val * 10) / 10  // 0.1 degree precision
    },
    mechanicalTilt: {
      type: 'number',
      required: false,
      min: 0,
      max: 10,
      description: 'Mechanical tilt angle',
      units: 'degrees'
    },
    txPower: {
      type: 'number',
      required: false,
      validators: [ThreeGPPValidators.validateTxPower],
      min: -130,
      max: 46,
      description: 'Transmit power',
      units: 'dBm'
    },
    crsGain: {
      type: 'number',
      required: false,
      min: -8,
      max: 15,
      description: 'Cell Reference Signal gain',
      units: 'dB'
    },
    beamWeights: {
      type: 'array',
      required: false,
      validators: [ThreeGPPValidators.validateBeamWeights],
      description: 'Beamforming weight vector'
    },
    ssbPeriodicity: {
      type: 'number',
      required: false,
      min: 5,
      max: 160,
      description: 'SSB periodicity',
      units: 'ms'
    }
  },
  customValidators: [
    // Cross-field validation: tilt + power consistency
    (data: CMParameters) => {
      if (data.electricalTilt !== undefined &&
        data.txPower !== undefined &&
        data.electricalTilt > 10 &&
        data.txPower > 43) {
        return {
          field: 'electricalTilt,txPower',
          message: 'High tilt + high power may cause excessive interference',
          constraint: 'Network planning best practice',
          value: { tilt: data.electricalTilt, power: data.txPower },
          expected: 'tilt <= 10 OR power <= 43'
        };
      }
      return null;
    }
  ]
};

/**
 * Recommendation validation schema
 */
export const RecommendationSchema: ValidationSchema<Recommendation> = {
  fields: {
    cellId: {
      type: 'string',
      required: true,
      description: 'Target cell identifier'
    },
    priority: {
      type: 'string',
      required: true,
      validators: [
        (value: string) => {
          const validPriorities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
          if (!validPriorities.includes(value)) {
            return {
              field: 'priority',
              message: 'Invalid priority level',
              constraint: 'Enum constraint',
              value,
              expected: validPriorities
            };
          }
          return null;
        }
      ]
    },
    recommendedAction: {
      type: 'object',
      required: true,
      description: 'Recommended CM parameter changes'
    },
    reasoning: {
      type: 'string',
      required: true,
      validators: [
        (value: string) => {
          if (value.length < 20) {
            return {
              field: 'reasoning',
              message: 'Reasoning must be at least 20 characters',
              constraint: 'Explainability requirement',
              value: value.length,
              expected: '>= 20 characters'
            };
          }
          return null;
        }
      ]
    },
    expectedGain: {
      type: 'object',
      required: true
    },
    similarPastSuccesses: {
      type: 'array',
      required: true,
      validators: [
        (value: any[]) => {
          if (value.length === 0) {
            return {
              field: 'similarPastSuccesses',
              message: 'Recommendation must be backed by at least one similar success',
              constraint: 'Evidence-based requirement',
              value: value.length,
              expected: '>= 1'
            };
          }
          return null;
        }
      ]
    },
    risks: {
      type: 'array',
      required: true
    },
    confidence: {
      type: 'number',
      required: true,
      min: 0,
      max: 1,
      validators: [
        (value: number) => {
          if (value < 0.3) {
            return {
              field: 'confidence',
              message: 'Confidence too low for automatic recommendation',
              constraint: 'Safety threshold',
              value,
              expected: '>= 0.3'
            };
          }
          return null;
        }
      ]
    }
  } as any
};

// ============================================================================
// Validator Engine
// ============================================================================

/**
 * Pydantic-style validator engine
 */
export class Validator<T> {
  private schema: ValidationSchema<T>;

  constructor(schema: ValidationSchema<T>) {
    this.schema = schema;
  }

  /**
   * Validate data against schema
   */
  validate(data: Partial<T>, options?: { coerce?: boolean }): ValidationResult<T> {
    const errors: ValidationError[] = [];
    const warnings: string[] = [];
    const validatedData: any = {};

    // Validate each field
    for (const [fieldName, schema] of Object.entries(this.schema.fields)) {
      const fieldSchema = schema as SchemaField;
      const value = (data as any)[fieldName];

      // Check required
      if (fieldSchema.required && value === undefined) {
        errors.push({
          field: fieldName,
          message: `Field '${fieldName}' is required`,
          constraint: 'required',
          value: undefined
        });
        continue;
      }

      // Skip if undefined and not required
      if (value === undefined) {
        continue;
      }

      // Coerce if enabled
      let processedValue = value;
      if (options?.coerce && fieldSchema.coerce) {
        processedValue = fieldSchema.coerce(value);
        if (processedValue !== value) {
          warnings.push(`Field '${fieldName}' coerced from ${value} to ${processedValue}`);
        }
      }

      // Type check
      const actualType = Array.isArray(processedValue) ? 'array' : typeof processedValue;
      if (actualType !== fieldSchema.type) {
        errors.push({
          field: fieldName,
          message: `Field '${fieldName}' has wrong type`,
          constraint: 'type',
          value: processedValue,
          expected: fieldSchema.type
        });
        continue;
      }

      // Range check for numbers
      if (fieldSchema.type === 'number') {
        if (fieldSchema.min !== undefined && processedValue < fieldSchema.min) {
          errors.push({
            field: fieldName,
            message: `Field '${fieldName}' below minimum`,
            constraint: 'min',
            value: processedValue,
            expected: `>= ${fieldSchema.min}`
          });
        }

        if (fieldSchema.max !== undefined && processedValue > fieldSchema.max) {
          errors.push({
            field: fieldName,
            message: `Field '${fieldName}' above maximum`,
            constraint: 'max',
            value: processedValue,
            expected: `<= ${fieldSchema.max}`
          });
        }
      }

      // Run custom validators
      if (fieldSchema.validators) {
        for (const validator of fieldSchema.validators) {
          const error = validator(processedValue, data);
          if (error) {
            errors.push(error);
          }
        }
      }

      validatedData[fieldName] = processedValue;
    }

    // Run schema-level custom validators
    if (this.schema.customValidators && errors.length === 0) {
      for (const validator of this.schema.customValidators) {
        const error = validator(validatedData as T);
        if (error) {
          errors.push(error);
        }
      }
    }

    const valid = errors.length === 0;

    return {
      valid,
      data: valid ? validatedData as T : undefined,
      errors,
      warnings: warnings.length > 0 ? warnings : undefined
    };
  }

  /**
   * Validate and throw on error
   */
  validateOrThrow(data: Partial<T>, options?: { coerce?: boolean }): T {
    const result = this.validate(data, options);

    if (!result.valid) {
      const errorMsg = result.errors
        .map(e => `${e.field}: ${e.message} (got ${e.value}, expected ${e.expected})`)
        .join('\n');

      throw new Error(`Validation failed:\n${errorMsg}`);
    }

    return result.data!;
  }
}

// ============================================================================
// Pre-configured Validators
// ============================================================================

export const cmParametersValidator = new Validator<CMParameters>(CMParametersSchema);
export const recommendationValidator = new Validator<Recommendation>(RecommendationSchema);

// ============================================================================
// Exports
// ============================================================================

// export {
//   Validator,
//   ThreeGPPValidators,
//   PhysicsValidators,
//   type ValidationError,
//   type ValidationResult,
//   type ValidationSchema,
//   type SchemaField,
//   type FieldValidator
// };
