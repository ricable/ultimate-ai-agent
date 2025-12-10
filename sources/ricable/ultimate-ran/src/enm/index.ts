/**
 * ENM MOM Schema & CM Writer - Main Export
 *
 * Ericsson ENM (Element Network Manager) integration module
 * Provides 3GPP-compliant MOM types, XML parsing, and CM operations
 */

// ============================================================================
// MOM Schema Exports
// ============================================================================

export {
  // Types
  ManagedElement,
  EUtranCellFDD,
  NRCellDU,
  AntennaUnit,
  SONParameters,

  // Zod Schemas
  ManagedElementSchema,
  EUtranCellFDDSchema,
  NRCellDUSchema,
  AntennaUnitSchema,
  SONParametersSchema,

  // Parser & Generator
  MOMXMLParser,
  MOMXMLGenerator,
  momParser,
  momGenerator,

  // Type Guards
  isEUtranCellFDD,
  isNRCellDU,
  isAntennaUnit,
} from './mom-schema.js';

// ============================================================================
// CM Writer Exports
// ============================================================================

export {
  // Types
  ManagedObjectType,
  CMOperation,
  CMChangeRequest,
  CMTransaction,
  CMWriteResult,
  ENMConnectionConfig,

  // Main Class
  ENMCMWriter,

  // Factory Functions
  createCMWriter,
  createBatchChanges,

  // Default Instance
  cmWriter,
} from './cm-writer.js';

// ============================================================================
// Re-export Examples (for testing/docs)
// ============================================================================

export * as examples from './enm-example.js';
